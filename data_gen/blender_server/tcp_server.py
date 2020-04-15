'''Source: https://github.com/fschr/simpletcp/tree/master/simpletcp'''

from scene_tools.blender_server.server_socket import ServerSocket

class TCPServer:
    # Mode specifies the IP address the server socket binds to.
    # mode can be one of two special values:
    # localhost -> (127.0.0.1)
    # public ->    (0.0.0.0)
    # otherwise, mode is interpreted as an IP address.
    # port specifies the port that the server socket binds to.
    # read_callback specifies the function that is called when
    # the server reads incoming data.
    # read_callback must be a function that takes three arguments:
    # The first argument must be a string which represents the IP
    # address that data was received from.
    # The second argument must be a queue (a queue.Queue()) which
    # is a tunnel of data to send to the socket that it receieved from.
    # The third argument must be data, which is a string of bytes
    # that the server received.
    def __init__(self, mode, port, read_callback,
                 maximum_connections=5, recv_bytes=2048):
        self.serversocket = ServerSocket(
            mode, port, read_callback, maximum_connections, recv_bytes
        )

    def run(self):
        self.serversocket.run()

    @property
    def ip(self):
        return self.serversocket.ip

    @property
    def port(self):
        return self.serversocket.port