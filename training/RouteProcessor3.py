import json
import os
import subprocess
import urllib.parse
import pandas as pd
import math
import time
def getHeadingBetweenPoints(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon)
    heading = math.degrees(math.atan2(y, x))
    heading = (heading + 360) % 360
    return heading

def runRouteProcessor(coords):
    useOnlineValhalla = False
    procCoordsNum = 70
    coordIdx = 0
    coordSize = len(coords)
    coordIter = 1
  


    combinedData = {"nodes":{}, "edges":{}, 
                    "edges_attributes":["start_node","end_node","length", "begin_heading", "end_heading", "speed", "speed_limit"],
                    "nodes_attributes":["lat", "lon", "range", "elevation", "edgeID", "percent_edge", "speed_limit", "speed_limit_mps", "edge_speed", "edge_speed_mps", "heading", "edge_mean_elevation", "edge_curvature", "edge_max_upward_grade", "edge_max_downward_grade", "edge_weighted_grade", "traffic_signal", "stop_sign", "yield_sign", "round_about", "edge_classification", "edge_link","intersection_type", "intersection_complexity", "way_id"]}
    cumRange = 0
    while coordIdx < coordSize - 1:
        locateJSONPayload = {
            "costing": "auto",
            "verbose": True,
            "locations": [],
        }
        traceJSONPayload = {
            "costing": "auto",
            "shape_match": "map_snap",
            "shape": []
        }
        elevationJSONPayload = {
            "range" : True,
            "shape" : []
        }

        while coordIdx < procCoordsNum*coordIter and coordIdx < coordSize :
            lon = coords[coordIdx][0]
            lat = coords[coordIdx][1]
            locDict = {"lat" : lat, "lon": lon}
            locateJSONPayload["locations"].append(locDict)
            traceJSONPayload["shape"].append(locDict)
            elevationJSONPayload["shape"].append(locDict)
            coordIdx += 1
        locateJSONPayloadURL = urllib.parse.quote(json.dumps(locateJSONPayload))
        traceJSONPayloadURL = urllib.parse.quote(json.dumps(traceJSONPayload))
        elevationJSONPayloadURL = urllib.parse.quote(json.dumps(elevationJSONPayload))


        while True:

            if useOnlineValhalla:
                locateURL = "https://valhalla1.openstreetmap.de/locate?json="
                traceAttributesURL = "https://valhalla1.openstreetmap.de/trace_attributes?json="
                elevationURL = "https://valhalla1.openstreetmap.de/height?json="
            else:
                locateURL = "http://localhost:8002/locate?json="
                traceAttributesURL = "http://localhost:8002/trace_attributes?json="
                elevationURL = "http://localhost:8002/height?json="

            locateCurlCmd = [
            "curl",
            "-X", "POST", locateURL + locateJSONPayloadURL]
            traceCurlCmd = [
            "curl",
            "-X", "POST", traceAttributesURL + traceJSONPayloadURL]
            elevationCurlCmd = [
            "curl",
            "-X", "POST", elevationURL + elevationJSONPayloadURL]

            locateResult = subprocess.run(locateCurlCmd, capture_output=True)
            traceResult = subprocess.run(traceCurlCmd, capture_output=True)
            elevationResult = subprocess.run(elevationCurlCmd, capture_output=True)

            # Handle subprocess errors and decode output with proper encoding
            try:
                if locateResult.returncode != 0:
                    print(f"Locate request failed with return code {locateResult.returncode}")
                    print(f"Error: {locateResult.stderr.decode('utf-8', errors='replace')}")
                    continue
                if traceResult.returncode != 0:
                    print(f"Trace request failed with return code {traceResult.returncode}")
                    print(f"Error: {traceResult.stderr.decode('utf-8', errors='replace')}")
                    continue
                if elevationResult.returncode != 0:
                    print(f"Elevation request failed with return code {elevationResult.returncode}")
                    print(f"Error: {elevationResult.stderr.decode('utf-8', errors='replace')}")
                    continue
                
                # Decode with UTF-8 and handle any encoding errors gracefully
                locateOutput = locateResult.stdout.decode('utf-8', errors='replace')
                traceOutput = traceResult.stdout.decode('utf-8', errors='replace')
                elevationOutput = elevationResult.stdout.decode('utf-8', errors='replace')
                
                if not locateOutput:
                    print("Locate request returned empty output")
                    continue
                if not traceOutput:
                    print("Trace request returned empty output")
                    continue
                if not elevationOutput:
                    print("Elevation request returned empty output")
                    continue
                
                locateCoordData = json.loads(locateOutput)
                traceCoordData = json.loads(traceOutput)
                elevationCoordData = json.loads(elevationOutput)

                if (any(point.get('type') == 'unmatched' for point in traceCoordData['matched_points'])) and (not useOnlineValhalla):
                    print("Unmatched points found in trace response. Retrying with different server endpoints...")
                    useOnlineValhalla = True
                    continue
                else:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                if useOnlineValhalla:
                    break
                else:
                    useOnlineValhalla = True
                    continue
            except Exception as e:
                print(f"Unexpected error processing coordinates: {e}")
                if useOnlineValhalla:
                    break
                else:
                    useOnlineValhalla = True
                    continue

        for idx, node in enumerate(locateCoordData):
            if coordIter > 1 and idx == 0:
                continue
            if coordIter > 1:
                glbIdx = idx + (coordIter - 1) * procCoordsNum - 1
            else:
                glbIdx = idx
            combinedData['nodes'][glbIdx]  = []
            combinedData['nodes'][glbIdx].append(node['input_lat'])
            combinedData['nodes'][glbIdx].append(node['input_lon'])
            combinedData['nodes'][glbIdx].append(elevationCoordData['range_height'][idx][0] + cumRange)
            combinedData['nodes'][glbIdx].append(elevationCoordData['range_height'][idx][1])
            if idx == 0:
                calcHeading = getHeadingBetweenPoints(node['input_lat'], node['input_lon'], 
                                                      locateCoordData[min(idx+1, len(locateCoordData)-1)]['input_lat'], 
                                                      locateCoordData[min(idx+1, len(locateCoordData)-1)]['input_lon'])
            else:
                        calcHeading = getHeadingBetweenPoints(locateCoordData[max(idx-1,0)]['input_lat'], 
                                                      locateCoordData[max(idx-1,0)]['input_lon'],
                                                      node['input_lat'], node['input_lon'])
            try:
                matchedEdgeIndex = traceCoordData['matched_points'][idx]['edge_index']
                edgeId = str(traceCoordData['edges'][matchedEdgeIndex]['id'])
            except IndexError:
                backIdx = 1
                while True:
                    try:
                        matchedEdgeIndex = traceCoordData['matched_points'][idx-backIdx]['edge_index']
                        edgeId = str(traceCoordData['edges'][matchedEdgeIndex]['id'])
                        break
                    except IndexError:
                        backIdx += 1
                        continue
            distanceAlongEdge = traceCoordData['matched_points'][idx]['distance_along_edge']
            edgeLength = traceCoordData['edges'][matchedEdgeIndex]['length']
            for edge in node['edges']:
                # if edge['edge']['stop_sign'] == True:
                #     print("Stop Sign at Node", idx, "Edge", str(edge['edge_id']['value']), "at way ID", edge['edge_info']['way_id'])
                try:
                    if edgeId == str(edge['edge_id']['value']) and ((abs(distanceAlongEdge*edgeLength*1000 - edge['percent_along']*edge['edge']['geo_attributes']['length']) < 5) or (abs(edge['percent_along'] -distanceAlongEdge) < 1e-1) or (abs(calcHeading - edge['heading']) < 10)):
                        combinedData['nodes'][glbIdx].append(edgeId)
                        combinedData['nodes'][glbIdx].append(edge['percent_along'])
                        if edge['edge_info']['speed_limit'] == 0 and glbIdx > 0 and combinedData['nodes'][glbIdx-1][combinedData['nodes_attributes'].index("speed_limit")] != 0:
                            combinedData['nodes'][glbIdx].append(combinedData['nodes'][glbIdx-1][combinedData['nodes_attributes'].index("speed_limit")])
                            combinedData['nodes'][glbIdx].append(combinedData['nodes'][glbIdx-1][combinedData['nodes_attributes'].index("speed_limit_mps")])
                        else:
                            speedlimit = edge['edge_info']['speed_limit']
                            speedlimit_mps = speedlimit * (10/36)
                            combinedData['nodes'][glbIdx].append(speedlimit)
                            combinedData['nodes'][glbIdx].append(speedlimit_mps)
                        edgespeed = edge['edge']['speeds']['default']
                        edgespeed_mps = edgespeed * (10/36)
                        combinedData['nodes'][glbIdx].append(edgespeed)
                        combinedData['nodes'][glbIdx].append(edgespeed_mps)
                        combinedData['nodes'][glbIdx].append(edge['heading'])
                        combinedData['nodes'][glbIdx].append(edge['edge_info']['mean_elevation'])
                        combinedData['nodes'][glbIdx].append(edge['edge']['geo_attributes']['curvature'])
                        combinedData['nodes'][glbIdx].append(edge['edge']['geo_attributes']['max_up_slope'])
                        combinedData['nodes'][glbIdx].append(edge['edge']['geo_attributes']['max_down_slope'])
                        combinedData['nodes'][glbIdx].append(edge['edge']['geo_attributes']['weighted_grade'])
                        combinedData['nodes'][glbIdx].append(int(edge['edge']['traffic_signal']))
                        combinedData['nodes'][glbIdx].append(int(edge['edge']['stop_sign']))
                        combinedData['nodes'][glbIdx].append(int(edge['edge']['yield_sign']))
                        combinedData['nodes'][glbIdx].append(int(edge['edge']['round_about']))
                        combinedData['nodes'][glbIdx].append(edge['edge']['classification']['classification'])
                        combinedData['nodes'][glbIdx].append(int(edge['edge']['classification']['link']))
                        if len(node['nodes']) > 0:
                            combinedData['nodes'][glbIdx][combinedData['nodes_attributes'].index("traffic_signal")] = int(node['nodes'][0]['traffic_signal'])
                            combinedData['nodes'][glbIdx].append(node['nodes'][0]['type'])
                            combinedData['nodes'][glbIdx].append(node['nodes'][0]['intersection_type'])
                        else:
                            combinedData['nodes'][glbIdx].append(None)
                            combinedData['nodes'][glbIdx].append(None)
                        combinedData['nodes'][glbIdx].append(str(edge['edge_info']['way_id']))
                        break
                except KeyError:
                    continue

            if (len(combinedData['nodes'][glbIdx]) < len(combinedData['nodes_attributes'])):
                if glbIdx > 0:
                    # print("Incomplete node data at index", glbIdx, "with length", len(combinedData['nodes'][glbIdx]), "expected length", len(combinedData['nodes_attributes']))
                    # print("Using previous nodes data to fill in missing attributes")
                    combinedData['nodes'][glbIdx] += combinedData['nodes'][glbIdx-1][combinedData['nodes_attributes'].index("edgeID"):]

        # print("Data retrieved upto coordinates", coordIdx, "out of", coordSize)
        coordIter += 1
        coordIdx -= 1 
        cumRange = combinedData['nodes'][coordIdx][combinedData['nodes_attributes'].index("range")]

    nodePD = pd.DataFrame.from_dict(combinedData['nodes'], orient='index', columns=combinedData['nodes_attributes'])
    return nodePD

if __name__ == "__main__":
    # downloadPath = os.getcwd()
    # fileName = "route.geojson"
    # fileName = os.path.join(downloadPath, fileName)

    # with open(fileName, "r") as coordJson:
    #     readFile = json.load(coordJson)

    # coords = readFile['geometry']['coordinates']
    # routeData = runRouteProcessor(coords)
    # routeData.to_csv("route_processed_data.csv", index=False)

    file_path = "C:\\Users\\vasud\\OneDrive\\Desktop\\HU\\HUProjPy311\\src\\0_00057.csv"
    try:
        df = pd.read_csv(file_path, low_memory=False)

        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            print(f"Skipping file: '{file_path}' - 'latitude' or 'longitude' columns missing.")

        # Extract coordinates in [[lon1, lat1], [lon2, lat2], ...] format
        coords = df[['longitude', 'latitude']].values.tolist()

        if not coords:
            print(f"Skipping file: '{file_path}' - No coordinates found.")

        # Call runRouteProcessor with the extracted coordinates
        nodePD = runRouteProcessor(coords)

        if nodePD.empty:
            print(f"Skipping file: '{file_path}' - runRouteProcessor returned empty data.")

        # Reset index of original DataFrame for proper merging
        df = df.reset_index(drop=True)

        # Combine the original DataFrame and the augmented DataFrame
        combined_df = pd.concat([df, nodePD], axis=1)

        # Save the combined DataFrame back to the original file path
        combined_df.to_csv(file_path, index=False)

        print(f"Successfully processed and saved: {file_path}")

    except FileNotFoundError:
        print(f"Skipping file: '{file_path}' - File not found.")
    except pd.errors.EmptyDataError:
        print(f"Skipping file: '{file_path}' - Empty file.")
    except pd.errors.ParserError as e:
        print(f"Skipping file: '{file_path}' - Parser error: {e}")
    except Exception as e:
        print(f"Skipping file: '{file_path}' - An unexpected error occurred: {e}")

print("Defined the 'process_csv_file' function.")