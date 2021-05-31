from threading import Lock, Thread
from time import sleep, time
import requests
import yaml
import os

class SendPolicy:
    def __init__(self) -> None:
        """
        Initialize SendPolicy module, establishing some vars.
        """
        self.ready = True
        self.mutex = Lock()

        with open(f'config{os.sep}config_insp.yaml', 'r') as f:
            self.vars = yaml.load(f)

        self.URL = self.vars['url_api']
        self.time = self.vars['delay']
        self.request_info = {
            'ts': self.get_now(),
            'location': self.vars['location']
        }
        self.upload_info = {
            'file_uploaded': None
        }


    def get_now(self) -> int:
        """
        Returns current floor time.

        Returns:
            int: timestamp current time
        """
        return int(time())


    def send(self, frame_in_bytes: bytes) -> None:
        """
        Start processing of send packet to API, verifying if is possible
        given the delay value.

        Args:
            frame_in_bytes (bytes): bytes to send as a file
        """
        if self.ready:
            self.ready = False
            exe = Thread(target=self.send2api, args=(frame_in_bytes,))
            exe.start()


    def send2api(self, frame_in_bytes: bytes) -> None:
        """
        Sends data to API. If not wearing mask, sends frame also.

        Args:
            frame_in_bytes (bytes): bytes to send as a file
        """
        # get current time
        self.request_info['ts'] = self.get_now()

        # build file structure
        if frame_in_bytes:
            self.upload_info['file_uploaded'] = (
                    'gotcha_frame.jpg',
                    frame_in_bytes
                )
        else:
            self.upload_info['file_uploaded'] = None

        # send req. / receive response to API
        response = requests.post(
                self.URL,
                data=self.request_info,
                files=self.upload_info
            )
        # print status
        print(response.status_code, response.json(), sep=' -|- ')
        # give a break of request! hehe
        self.wait_until_ready(self.time)


    def wait_until_ready(self, time2wait):
        """
        Sleeps for <time2wait> seconds

        Args:
            time2wait (int): Seconds to wait
        """
        sleep(time2wait)
        self.ready = True