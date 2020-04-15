#Example code for client. Don't use for anything else

from scene_tools.blender_server.client_socket import ClientSocket
from scene_tools.blender_server.config import  *

s1 = ClientSocket("localhost", PORT)
response = s1.send("")

s2 = ClientSocket(IP_ADDRESS, PORT, single_use=False)
r1 = s2.send("Hello for the first time...")
r2 = s2.send("...and hello for the last!")
s2.close()

# Display the correspondence
print("s1 sent\t\tHello, World!")
print("s1 received\t\t{}".format(response.decode("UTF-8")))
print("-------------------------------------------------")
print("s2 sent\t\tHello for the first time....")
print("s2 received\t\t{}".format(r1.decode("UTF-8")))
print("s2 sent\t\t...and hello for the last!.")
print("s2 received\t\t{}".format(r2.decode("UTF-8")))

if __name__ == "__main__":
    pass