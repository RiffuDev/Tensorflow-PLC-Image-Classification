import asyncio
import time
from asyncua import Client, ua

async def opcuatest():
    print("run")
    async with Client(url='opc.tcp://192.168.11.47:4840') as client:
        b = True
        while True:
            # Do something with client
            node = client.get_node('ns=4;i=17')
            value = await node.read_value()
            print(value)
            try:
                client_node_dv = ua.DataValue(ua.Variant(b, ua.VariantType.Boolean))
                await node.set_value(client_node_dv)
                time.sleep(1.5)
                b = not b
            except:
                print("Error Occurred")
            
async def opcuaMain():
    task = asyncio.create_task (opcuatest())
    await task

asyncio.run(opcuaMain())
