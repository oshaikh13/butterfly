import logging
import os
import socket
import tornado
import tornado.gen
import tornado.web
import tornado.websocket
import numpy as np
import cv2
import StringIO
import zlib
import sys
import traceback
import rh_logger
import settings
from requestparser import RequestParser
from urllib2 import HTTPError


class WebServerHandler(tornado.web.RequestHandler):

    def initialize(self, webserver):
        self._webserver = webserver

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self, uri):
        '''
        '''
        self._webserver.handle(self)


class WebServer:

    def __init__(self, core, port=2001):
        '''
        '''
        self._core = core
        self._port = port

    def start(self):
        '''
        '''

        ip = socket.gethostbyname('')
        port = self._port

        webapp = tornado.web.Application([

            (r'/metainfo/(.*)', WebServerHandler, dict(webserver=self)),
            (r'/data/(.*)', WebServerHandler, dict(webserver=self)),
            # (r'/(.*)', tornado.web.StaticFileHandler,
            #  dict(path=os.path.join(os.path.dirname(__file__),'../web'),
            #       default_filename='index.html'))

        ])

        webapp.listen(port, max_buffer_size=1024 * 1024 * 150000)
        startup_msg = ('Starting webserver at \033[93mhttp://' + ip + ':' +
                       str(port) + '\033[0m')

        rh_logger.logger.report_event(startup_msg)

        tornado.ioloop.IOLoop.instance().start()

    @tornado.gen.coroutine
    def handle(self, handler):
        '''
        '''
        content = None

        splitted_request = handler.request.uri.split('/')

        if splitted_request[1] == 'metainfo':

            # content = self._core.get_meta_info(path)
            content = 'metainfo'
            handler.set_header("Content-type", 'text/html')

        # image data request
        elif splitted_request[1] == 'data':

            try:
                parser = RequestParser()
                args = parser.parse(splitted_request[2:])

                # Call the cutout method
                volume = self._core.get(*args)

                # Check if we got nothing in the case of a request outside the
                # data with fit=True
                if volume.size == 0:
                    raise IndexError('Tile index out of bounds')

                # Color mode is equivalent to segmentation color request right
                # now
                color = parser.optional_queries[
                    'segcolor'] and parser.optional_queries['segmentation']

                # Accepted image output formats
                image_formats = settings.SUPPORTED_IMAGE_FORMATS

                # Process output
                out_dtype = np.uint8
                output_format = parser.output_format

                if output_format == 'zip' and not color:
                    # Rotate out of numpy array
                    volume = volume.transpose(1, 0, 2)
                    zipped_data = zlib.compress(
                        volume.astype(out_dtype).tostring('F'))

                    output = StringIO.StringIO()
                    output.write(zipped_data)
                    content = output.getvalue()
                    content_type = 'application/octet-stream'
                elif output_format in image_formats:
                    if color:
                        volume = volume[:, :, :, [2, 1, 0]]
                        content = cv2.imencode(
                            '.' + output_format,
                            volume[
                                :,
                                :,
                                0,
                                :].astype(out_dtype))[1].tostring()
                    else:
                        content = cv2.imencode(
                            '.' + output_format,
                            volume[
                                :,
                                :,
                                0].astype(out_dtype))[1].tostring()
                    content_type = 'image/' + output_format
                else:
                    raise HTTPError(handler.request.uri,
                                    400,
                                    'Output file format not supported',
                                    [], None)

                # Show some basic statistics
                
                rh_logger.logger.report_event(
                    'Total volume shape: %s' % str(volume.shape))
                handler.set_header('Content-Type', content_type)

            except (KeyError, ValueError):
                rh_logger.logger.report_error('Missing query',
                                              log_level=logging.WARNING)
                content = 'Error 400: Bad request<br>Missing query'
                content_type = 'text/html'
                handler.set_status(400)
            except IndexError:
                rh_logger.logger.report_exception(msg='Could not load image')
                content = 'Error 400: Bad request<br>Could not load image'
                content_type = 'text/html'
                handler.set_status(400)
            except HTTPError, http_error:
                content = http_error.msg
                if len(http_error.hdrs) == 0:
                    handler.set_header('Content-Type', "text/html")
                handler.set_status(http_error.code)
            # except Exception:
            #   traceback.print_exc(file=sys.stdout)

        # invalid request
        if not content:
            handler.set_status(404)
            content = 'Error 404: Not found'
            handler.set_header("Content-Type", 'text/html')

        # handler.set_header('Cache-Control',
        #                    'no-cache, no-store, must-revalidate')
        # handler.set_header('Pragma','no-cache')
        # handler.set_header('Expires','0')
        handler.set_header('Access-Control-Allow-Origin', '*')

        # Temporary check for img output
        handler.write(content)