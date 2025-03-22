import psutil
import pandas as pd
import matplotlib.pyplot as plt



print("Hello, welcom in program for information about your system")

print("options: ")
print("1) Information about network interfaces")
print("2) Information about network connections")
print("3) Information about packet send")
print("4) Information about network speed")
print("5) Graphs")


user_input = int(input("Your choice: "))

if user_input == 1:

    connections = psutil.net_if_addrs()


    data_interface = []
    for interface, addrs in connections.items():
        for conn in addrs: 
            data_interface.append({
                "interface": interface,           
                "family": conn.family.name,       
                "address": conn.address,         
                "netmask": conn.netmask,          
                "broadcast": conn.broadcast       
            })
    print(" ")


    df_1 = pd.DataFrame(data_interface)
    print(df_1.head())
    df_1.to_csv("net_connections.csv", index=False)


    print(" ")


if user_input == 2:


    connections_network = psutil.net_connections()

    # Převedení dat na seznam slovníků
    data = []
    for conn in connections_network:#ineruji přes kažkdy objekt 
        data.append({
            "Local Address": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "N/A",
            "Remote Address": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "N/A",
            "Status": conn.status,
            "PID": conn.pid
        })

    df = pd.DataFrame(data)
    print(df.head())
    df.to_csv("net_connections.csv", index=False)




print(" ")

if user_input == 3:
    connections_interface = psutil.net_io_counters(pernic=True)


    data_interface_comunication = []
    for interface, stats in connections_interface.items():  
        data_interface_comunication.append({
                "interface": interface, 
                "bytes_sent": stats.bytes_sent,           
                "bytes_recv": stats.bytes_recv,       
                "packets_sent": stats.packets_sent,         
                "packets_recv": stats.packets_recv,          
                "errin": stats.errin,
                "errout": stats.errout, 
                "dropin": stats.dropin,
                "dropout": stats.dropout      
        })

    df_3 = pd.DataFrame(data_interface_comunication)
    print(df_3.head())
    df_3.to_csv("net_connections.csv", index=False)

if user_input == 4:


    print(" ")
    connections_interface_speed = psutil.net_if_stats()


    data_interface_comunication = []
    for interface, stats in connections_interface_speed.items():  
        data_interface_comunication.append({
                "interface": interface, 
                "isup": stats.isup,           
                "duplex": stats.duplex,       
                "speed": stats.speed,         
                "mtu": stats.mtu,          
                "flags": stats.flags,

        })

    df_4 = pd.DataFrame(data_interface_comunication)
    print(df_4.head())
    df_3.to_csv("net_connections.csv", index=False)






