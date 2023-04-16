import socket

def set_ap(nazov_siete, heslo_siete):

    my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    my_socket.bind(('',8889))
    cmd_str = 'command'
    print ('sending command %s' % cmd_str)
    my_socket.sendto(cmd_str.encode('utf-8'), ('192.168.10.1', 8889))
    response, ip = my_socket.recvfrom(100)
    print('from %s: %s' % (ip, response))

    cmd_str = 'ap %s %s' % (nazov_siete,heslo_siete)
    print('sending command %s' % cmd_str)
    my_socket.sendto(cmd_str.encode('utf-8'), ('192.168.10.1', 8889))
    response, ip = my_socket.recvfrom(100)
    print('from %s: %s' %(ip, response))

set_ap('janda', '24688642')
