from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, InvalidSessionIdException
from selenium.webdriver.chrome.service import Service   
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains

import os
import time

class MapDisplayHandler:
    """MapDisplayHandler that resets driver timeout whenever the user interacts (click/keydown/touch).

    It injects a small JS snippet into the page which updates `window.__lastUserInteraction` (ms since epoch)
    on user events. The Python handler polls that value and uses it to reset the timeout, so the browser
    won't quit as long as the user keeps interacting.
    """
    def __init__(self, downloadPath=None, fileName="route.geojson", driverTimeout=240,
                 open_url="https://www.openstreetmap.org/directions", debug_print_interval=5,
                 url_wait_timeout=1, directions_sleep=2, download_sleep=2, poll_interval=0.5):
        """Create handler.

        Args:
            downloadPath: directory to save downloads (defaults to cwd).
            fileName: name of downloaded route file (default 'route.geojson').
            driverTimeout: inactivity timeout in seconds (default 240).
            open_url: initial URL to open in the browser.
            debug_print_interval: seconds between debug prints (default 5).
            url_wait_timeout: timeout for waiting URL change (default 1).
            directions_sleep: seconds to wait after acquiring directions (default 2).
            download_sleep: seconds to wait after clicking download (default 2).
            poll_interval: seconds to sleep between run loop iterations (default 0.5).
        """
        if downloadPath is None:
            downloadPath = os.getcwd()
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_experimental_option("prefs", {
            "download.default_directory": downloadPath,
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True
        })
        self.fileName = fileName
        self.fileName = os.path.join(downloadPath, self.fileName)
        self._listener_injected = False
        self.driverTimeout = driverTimeout
        self.open_url = open_url
        self.debug_print_interval = debug_print_interval
        self.url_wait_timeout = url_wait_timeout
        self.directions_sleep = directions_sleep
        self.download_sleep = download_sleep
        self.poll_interval = poll_interval

    def initDriver(self):
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.get(self.open_url)
        # initialize timing
        self.driverStartTime = time.time()
        self.lastInteractionTime = self.driverStartTime
        self.initURL = self.driver.current_url
        self.directionsAcquired = False
        self.driveElapsedTime = 0
        self.userClosedBrowser = False
        # try to inject listener immediately (may fail until page loads)
        try:
            self._inject_interaction_listener()
        except Exception:
            # we'll retry injection later in run loop
            self._listener_injected = False

    def _inject_interaction_listener(self):
        """Inject JS on the page to record last user interaction timestamp.

        The JS sets `window.__lastUserInteraction` to Date.now() on several events.
        """
        if not hasattr(self, 'driver'):
            return
        js = (
            "window.__lastUserInteraction = window.__lastUserInteraction || Date.now();"
            "(function(){"
            "  try{"
            "    var events = ['click','keydown','touchstart','mousedown','mouseup'];"
            "    events.forEach(function(ev){"
            "      try{ document.addEventListener(ev, function(){ window.__lastUserInteraction = Date.now(); }, true); }catch(e){}"
            "      try{ for(var i=0;i<window.frames.length;i++){ try{ window.frames[i].document.addEventListener(ev, function(){ window.__lastUserInteraction = Date.now(); }, true); }catch(e){} } }catch(e){}"
            "    });"
            "    document.addEventListener('visibilitychange', function(){ if(!document.hidden) window.__lastUserInteraction = Date.now(); }, true);"
            "    window.__interactionListenerActive = true;"
            "  }catch(e){ window.__interactionListenerActive = false; }"
            "})();"
        )
        # execute_script can fail if page not ready; caller should handle
        self.driver.execute_script(js)
        self._listener_injected = True

    def _get_last_user_interaction(self):
        """Return tuple (timestamp_seconds, active_flag) or (None, False) on error.

        `timestamp_seconds` is POSIX float (seconds) or None. `active_flag` indicates
        whether the injected listener reports active.
        """
        try:
            res = self.driver.execute_script(
                'return {ts: (window.__lastUserInteraction || 0), active: !!window.__interactionListenerActive};'
            )
            if not isinstance(res, dict):
                return (None, False)
            ts = res.get('ts', 0)
            active = bool(res.get('active', False))
            try:
                ts_s = float(ts) / 1000.0
            except Exception:
                ts_s = None
            return (ts_s, active)
        except Exception:
            return (None, False)

    def runMapDisplayHandler(self) :
        try:
            # If listener not yet injected (e.g., first load), try again
            if not self._listener_injected:
                try:
                    self._inject_interaction_listener()
                except Exception:
                    pass

            # update lastInteractionTime from page if available
            last_js, listener_active = self._get_last_user_interaction()
            if last_js is not None and last_js and last_js > 0:
                self.lastInteractionTime = last_js

            self.driveElapsedTime = time.time() - self.lastInteractionTime
            # debug: print remaining time every `debug_print_interval` seconds
            if not hasattr(self, '_last_debug_print'):
                self._last_debug_print = 0
            if time.time() - self._last_debug_print >= self.debug_print_interval:
                time_left = max(0.0, self.driverTimeout - self.driveElapsedTime)
                raw_ts = last_js if last_js is not None else 0
                print(f"Time left until timeout due to inactivity: {time_left:.1f}s | listener_active={listener_active} | last_js_ts={raw_ts}")
                self._last_debug_print = time.time()

            if self.driveElapsedTime > self.driverTimeout:
                print("Driver timed out due to inactivity")
                try:
                    self.driver.quit()
                except Exception:
                    pass
                return

            # Remember current URL and wait for it to change (user submitted directions)
            self.initURL = self.driver.current_url
            self.directionsAcquired = False

            try:
                # wait a short time for URL change (short timeout so loop keeps polling and printing)
                WebDriverWait(self.driver, self.url_wait_timeout).until(lambda _: self.driver.current_url != self.initURL)
            except TimeoutException:
                # timed out waiting for URL change — loop will continue and check inactivity
                return
            except WebDriverException as e:
                # ignore transient webdriver issues
                print("WebDriver exception while waiting for URL change:", e)
                return

            # At this point the page likely changed; re-inject listener
            try:
                self._inject_interaction_listener()
            except Exception:
                pass

            # Try to find direction elements
            try:
                directionsText = self.driver.find_element(By.CSS_SELECTOR, 'h2.me-4.text-break')
                directionsRoute = self.driver.find_element(By.CSS_SELECTOR, 'div#directions_route')
                directionsError = self.driver.find_element(By.CSS_SELECTOR, 'div#directions_error')
            except Exception:
                # elements not found — continue monitoring
                self.initURL = self.driver.current_url
                return

            time.sleep(self.directions_sleep)
            if directionsText.text == "Directions" and directionsRoute.is_displayed():
                self.directionsAcquired = True
                print("Got new directions!")
            elif directionsText.text == "Directions" and directionsError.is_displayed():
                print("Failed in direction fetching")
                self.initURL = self.driver.current_url
                return
            else:
                self.initURL = self.driver.current_url

            if self.directionsAcquired:
                if os.path.exists(self.fileName):
                    os.remove(self.fileName)
                directionsDownload = self.driver.find_element(By.CSS_SELECTOR, 'a#directions_route_download')
                # time.sleep(2)  # small delay to ensure file is ready for download
                directionsDownload.click()
                time.sleep(self.download_sleep)

        except InvalidSessionIdException:
            print("User closed the browser, exiting script.")
            self.userClosedBrowser = True
            return

if __name__ == "__main__":
    myMapHandler = MapDisplayHandler()
    myMapHandler.initDriver()
    # simple loop to keep running until closed or timeout
    try:
        while not getattr(myMapHandler, 'userClosedBrowser', False):
            myMapHandler.runMapDisplayHandler()
            time.sleep(0.5)
    except KeyboardInterrupt:
        try:
            myMapHandler.driver.quit()
        except Exception:
            pass
