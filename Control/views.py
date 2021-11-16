from django.shortcuts import render
import socket
import serial
from django.views.decorators.csrf import csrf_exempt,csrf_protect
from .forms import KeyForm
import json

# for device in devices:
#     print(device)


def control_view(request):
    def netcat(content):
        clientsocket.send(content.encode())


    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #
    # print(socket.gethostbyname(socket.gethostname()))
    #
    # #s.bind(('192.168.68.102', 5050))
    # s.bind(('127.0.0.1', 5050))
    #
    # print(socket.gethostbyname(socket.gethostname()))
    # #s.bind(('10.81.15.219', 5050))
    #
    #
    #
    # s.listen(1)
    #
    # clientsocket , address = s.accept()
    #
    # print("passe")

    #while True:  # making a loop

    # form = KeyForm(request.POST)
    #
    # if form.is_valid():
    key = request.POST.get('key')


    print("Key" + ": " + str(key))


    return render(request,  "controlWindow.html")

#@csrf_exempt
def test_view(request):
    if request.method == 'POST':
        print("yes")
        body_unicode = request.body.decode('utf-8')
        print(body_unicode)
        # received_json = json.loads(body_unicode)

    # x = json.loads(request.body.decode('utf-8'))
    # print(x)

    return render(request,  "testWindow.html")