"""Faster route processor using parallel Valhalla requests.

Improvements over RouteProcessor3:
- Uses `requests` instead of subprocess/curl (lower overhead).
- Splits coordinates into overlapping chunks and processes chunks in parallel
  with a ThreadPoolExecutor.
- Per-chunk: sends `locate`, `trace_attributes`, and `height` requests (concurrently)
  and retries on failures with an online fallback.

Notes:
- This assumes the Valhalla server can accept multiple concurrent requests.
- Install `requests` if missing: `pip install requests`.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import math
import time
import pandas as pd
import numpy as np

# Tunable parameters
PROC_COORDS_NUM = 70
CHUNK_STEP = PROC_COORDS_NUM - 1  # overlap 1 to mimic original behavior
REQUEST_TIMEOUT = 15.0  # seconds
MAX_WORKERS = 8


def getHeadingBetweenPoints(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon))
    heading = math.degrees(math.atan2(y, x))
    heading = (heading + 360) % 360
    return heading


def _post_valhalla(base_url, endpoint, payload, timeout=REQUEST_TIMEOUT):
    url = base_url.rstrip('/') + '/' + endpoint
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise


def _process_chunk(chunk_coords, chunk_index, use_online=False, max_retries=2, request_timeout=REQUEST_TIMEOUT,
                   local_base='http://localhost:8002', online_base='https://valhalla1.openstreetmap.de', per_chunk_parallel=3):
    """Process a single chunk: call locate, trace_attributes, height.

    Returns tuple (chunk_index, locate, trace, elevation).
    """

    locate_payload = {"costing": "auto", "verbose": True, "locations": []}
    trace_payload = {"costing": "auto", "shape_match": "map_snap", "shape": []}
    elev_payload = {"range": True, "shape": []}

    for lon, lat in chunk_coords:
        loc = {"lat": lat, "lon": lon}
        locate_payload['locations'].append(loc)
        trace_payload['shape'].append(loc)
        elev_payload['shape'].append(loc)

    attempts = 0
    last_exc = None
    while attempts <= max_retries:
        attempts += 1
        for base in (local_base, online_base) if not use_online else (online_base,):
            try:
                # send three requests in parallel for lower latency
                with ThreadPoolExecutor(max_workers=per_chunk_parallel) as tex:
                    f_loc = tex.submit(_post_valhalla, base, 'locate?json=', locate_payload, request_timeout)
                    f_trace = tex.submit(_post_valhalla, base, 'trace_attributes?json=', trace_payload, request_timeout)
                    f_elev = tex.submit(_post_valhalla, base, 'height?json=', elev_payload, request_timeout)
                    locate = f_loc.result()
                    trace = f_trace.result()
                    elev = f_elev.result()

                # detect unmatched matched_points as previous code did
                if any(point.get('type') == 'unmatched' for point in trace.get('matched_points', [])) and base != online_base:
                    # try online in next iteration
                    use_online = True
                    continue

                return (chunk_index, locate, trace, elev)
            except Exception as e:
                last_exc = e
                # try next base (online) or retry
                continue

    # If we get here, all attempts failed
    raise RuntimeError(f"Chunk {chunk_index} failed after {attempts} attempts: {last_exc}")


def runRouteProcessor(coords, proc_coords_num=PROC_COORDS_NUM, max_workers=MAX_WORKERS,
                      request_timeout=REQUEST_TIMEOUT, local_base='http://localhost:8002',
                      online_base='https://valhalla1.openstreetmap.de', per_chunk_parallel=3, max_retries=2):
    """Main entrypoint. Returns a DataFrame similar to RouteProcessor3.runRouteProcessor.

    Additional tunables: request_timeout, local_base, online_base, per_chunk_parallel, max_retries.
    """
    if not coords:
        return pd.DataFrame()

    # Build overlapping chunks
    chunks = []
    N = len(coords)
    step = max(1, proc_coords_num - 1)
    idx = 0
    chunk_id = 0
    while idx < N:
        end = min(idx + proc_coords_num, N)
        chunk = coords[idx:end]
        chunks.append((chunk_id, idx, chunk))
        chunk_id += 1
        if end >= N:
            break
        idx += step

    results = [None] * len(chunks)

    # Parallel processing of chunks
    with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as ex:
        futures = {ex.submit(_process_chunk, ch[2], ch[0], False, max_retries, request_timeout, local_base, online_base, per_chunk_parallel): ch for ch in chunks}
        for fut in as_completed(futures):
            ch = futures[fut]
            try:
                chunk_index, locate, trace, elev = fut.result()
                results[chunk_index] = (locate, trace, elev, ch[1])  # include original start idx
            except Exception as e:
                raise

    # Assemble combinedData ordered by chunk start index
    combined_nodes = {}
    combined_attrs = [
        'lat', 'lon', 'range', 'elevation', 'edgeID', 'percent_edge', 'speed_limit', 'speed_limit_mps',
        'edge_speed', 'edge_speed_mps', 'heading', 'edge_mean_elevation', 'edge_curvature',
        'edge_max_upward_grade', 'edge_max_downward_grade', 'edge_weighted_grade', 'traffic_signal',
        'stop_sign', 'yield_sign', 'round_about', 'edge_classification', 'edge_link', 'intersection_type',
        'intersection_complexity', 'way_id'
    ]

    # Helper to extract per-node info (following logic from RP3)
    node_idx_global = 0
    cumRange = 0
    for res in results:
        if res is None:
            continue
        locate, trace, elev, chunk_start_idx = res
        # locate may be a list of node dicts (same shape as before)
        for idx_in_chunk, node in enumerate(locate):
            glbIdx = chunk_start_idx + idx_in_chunk
            entry = []
            entry.append(node.get('input_lat'))
            entry.append(node.get('input_lon'))
            # elevation: elevation service returns range_height list per point
            try:
                elev_point = elev['range_height'][idx_in_chunk]
                range_val = elev_point[0] + cumRange
                elev_val = elev_point[1]
            except Exception:
                range_val = None
                elev_val = None
            entry.append(range_val)
            entry.append(elev_val)

            # heading calculation
            if idx_in_chunk == 0:
                # use next point if available
                nxt = locate[min(idx_in_chunk + 1, len(locate) - 1)]
                calcHeading = getHeadingBetweenPoints(node.get('input_lat'), node.get('input_lon'), nxt.get('input_lat'), nxt.get('input_lon'))
            else:
                prev = locate[max(idx_in_chunk - 1, 0)]
                calcHeading = getHeadingBetweenPoints(prev.get('input_lat'), prev.get('input_lon'), node.get('input_lat'), node.get('input_lon'))

            # Determine matched edge and edgeId similar to RP3 logic
            try:
                matchedEdgeIndex = trace['matched_points'][idx_in_chunk]['edge_index']
                edgeId = str(trace['edges'][matchedEdgeIndex]['id'])
            except Exception:
                # fallback: search backward
                backIdx = 1
                found = False
                while True:
                    try:
                        matchedEdgeIndex = trace['matched_points'][idx_in_chunk - backIdx]['edge_index']
                        edgeId = str(trace['edges'][matchedEdgeIndex]['id'])
                        found = True
                        break
                    except Exception:
                        backIdx += 1
                        if backIdx > idx_in_chunk:
                            break
                if not found:
                    edgeId = None

            # extract data similar to original loop
            distanceAlongEdge = None
            edgeLength = None
            try:
                distanceAlongEdge = trace['matched_points'][idx_in_chunk].get('distance_along_edge')
                edgeLength = trace['edges'][matchedEdgeIndex].get('length')
            except Exception:
                pass

            # find matching edge from node['edges'] list
            matched = False
            for edge in node.get('edges', []):
                try:
                    if edgeId is None:
                        continue
                    if edgeId == str(edge['edge_id']['value']) and ((abs(distanceAlongEdge*edgeLength*1000 - edge['percent_along']*edge['edge']['geo_attributes']['length']) < 5) or (abs(edge['percent_along'] -distanceAlongEdge) < 1e-1) or (abs(calcHeading - edge['heading']) < 10)):
                        entry.append(edgeId)
                        entry.append(edge.get('percent_along'))
                        if edge['edge_info'].get('speed_limit', 0) == 0 and glbIdx > 0 and (combined_nodes.get(glbIdx - 1)):
                            prev_speed_limit = combined_nodes[glbIdx - 1][0][combined_attrs.index('speed_limit')]
                            prev_speed_limit_mps = combined_nodes[glbIdx - 1][0][combined_attrs.index('speed_limit_mps')]
                            entry.append(prev_speed_limit)
                            entry.append(prev_speed_limit_mps)
                        else:
                            speedlimit = edge['edge_info'].get('speed_limit', 0)
                            speedlimit_mps = speedlimit * (10 / 36)
                            entry.append(speedlimit)
                            entry.append(speedlimit_mps)

                        edgespeed = edge['edge']['speeds'].get('default', 0)
                        edgespeed_mps = edgespeed * (10 / 36)
                        entry.append(edgespeed)
                        entry.append(edgespeed_mps)
                        entry.append(edge.get('heading'))
                        entry.append(edge['edge_info'].get('mean_elevation'))
                        entry.append(edge['edge']['geo_attributes'].get('curvature'))
                        entry.append(edge['edge']['geo_attributes'].get('max_up_slope'))
                        entry.append(edge['edge']['geo_attributes'].get('max_down_slope'))
                        entry.append(edge['edge']['geo_attributes'].get('weighted_grade'))
                        entry.append(int(edge['edge'].get('traffic_signal', False)))
                        entry.append(int(edge['edge'].get('stop_sign', False)))
                        entry.append(int(edge['edge'].get('yield_sign', False)))
                        entry.append(int(edge['edge'].get('round_about', False)))
                        entry.append(edge['edge']['classification'].get('classification'))
                        entry.append(int(edge['edge']['classification'].get('link', 0)))
                        if node.get('nodes'):
                            # ensure space reserved for traffic_signal index if previous entries exist
                            if len(entry) <= combined_attrs.index('traffic_signal'):
                                # pad
                                entry += [None] * (combined_attrs.index('traffic_signal') - len(entry) + 1)
                            entry[combined_attrs.index('traffic_signal')] = int(node['nodes'][0].get('traffic_signal', 0))
                            entry.append(node['nodes'][0].get('type'))
                            entry.append(node['nodes'][0].get('intersection_type'))
                        else:
                            entry.append(None)
                            entry.append(None)
                        entry.append(str(edge['edge_info'].get('way_id')))
                        matched = True
                        break
                except Exception:
                    continue

            if not matched:
                # fill with placeholders to keep consistent length
                entry += [None] * (len(combined_attrs) - len(entry))

            # append entry to combined_nodes (allow duplicates)
            combined_nodes.setdefault(glbIdx, []).append(entry)
            # update node counter
            node_idx_global += 1
        # update cumRange to last processed range if available
        try:
            last_idx = max(combined_nodes.keys())
            last_range = combined_nodes[last_idx][combined_attrs.index('range')]
            if last_range is not None:
                cumRange = last_range
        except Exception:
            pass

    # Merge duplicated entries (same glbIdx may have multiple candidate entries)
    merged_nodes = {}
    for glbIdx in sorted(combined_nodes.keys()):
        entries = combined_nodes[glbIdx]
        if not entries:
            continue
        if len(entries) == 1:
            merged = entries[0]
        else:
            # merge column-wise: for 'range' take max of non-null, otherwise take first non-null
            merged = []
            for col in range(len(combined_attrs)):
                vals = [e[col] for e in entries if (col < len(e) and e[col] is not None)]
                if not vals:
                    merged.append(None)
                else:
                    if combined_attrs[col] == 'range':
                        # choose maximum range
                        try:
                            numvals = [float(v) for v in vals if v is not None]
                            merged.append(max(numvals) if numvals else None)
                        except Exception:
                            merged.append(vals[0])
                    else:
                        merged.append(vals[0])
        # ensure correct length
        if len(merged) < len(combined_attrs):
            merged += [None] * (len(combined_attrs) - len(merged))
        merged_nodes[glbIdx] = merged

    # Build DataFrame from merged nodes
    nodePD = pd.DataFrame.from_dict(merged_nodes, orient='index', columns=combined_attrs)

    # Sanity check: forward-fill any unfilled/missing values using previous row's values.
    # Ensure 'range' is numeric, fill gaps, and if a decrease is detected at row i
    # (i.e. range[i] < range[i-1]) then add range[i-1] to range[i] and to all
    # subsequent rows as requested.
    if not nodePD.empty:
        nodePD = nodePD.sort_index()
        # coerce to numeric and fill simple gaps
        # nodePD['range'] = pd.to_numeric(nodePD['range'], errors='coerce')
        # nodePD['range'] = nodePD['range'].ffill().fillna(0).astype(float)

        # Apply additive monotonic fix: when a drop is found, add previous value to
        # current and all following rows (and continue scanning).
        # make a writable copy of the series values
        arr = nodePD['range'].to_numpy(dtype=float, copy=True)
        n = arr.size
        # iterate from bottom to top so adjustments propagate upwards first
        for i in range(n - 1, 0, -1):
            if arr[i] < arr[i - 1]:
                delta = arr[i - 1]
                arr[i:] = arr[i:] + delta

        nodePD['range'] = arr

        # forward-fill remaining columns and replace any remaining NaNs with 0
        nodePD = nodePD.ffill()
        nodePD = nodePD.fillna(0)

    return nodePD


if __name__ == '__main__':
    print('RouteProcessor4 loaded — use runRouteProcessor(coords)')
