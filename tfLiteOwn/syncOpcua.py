# import sys
# sys.path.insert(0, "../..")

import time

from asyncua.sync import Client, ua

HOME_POS_LEFT = "ns=4;i=12"
HOME_POS_RIGHT = "ns=4;i=15"
START_CYL = "ns=4;i=18"

def connect(url):
    try:
        with Client(url) as client:
            print("Connected")
            pickCyl(client)
            return True, client

    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
        return False, None
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, None
    
def readNode(client, nodeAddr):
    node = client.get_node(nodeAddr)
    data = node.read_value()
    return data


def setNode(client, nodeAddr, data):
     
    node = client.get_node(nodeAddr)
    client_node_dv = ua.DataValue(ua.Variant(data, ua.VariantType.Int16))
    node.set_value(client_node_dv) 


def pickCyl(client):

    # data = readNode(client, HOME_POS_LEFT)
    # print(f"isHOME: {data}")

    # if(data):

        setNode(client, START_CYL, 2)
        time.sleep(0.5)
        setNode(client, START_CYL, 0)



if __name__ == "__main__":
     
    # isconnected, client = connect("opc.tcp://192.168.11.47:4840") 
    # if(isconnected):
    #     print("Connected")
    #     pickCyl(client)
    # else:
    #     print("Not connected")
    isC, client = connect("opc.tcp://192.168.11.47:4840")
    print(f"IS: {isC}")

    # if(isC):
    #     pickCyl(client)
