import sys
import zmq
import cPickle as pickle


address = 'tcp://localhost:'

senti_messages = ['i hate everything because it sucks :(', 
                  'i love my iphone because apple is the best :)',
                  'the reporter was completely impartial as am i']

def test_annotator(socket):
    while True:
        for txt in senti_messages:
            print('Sending Request: %s' % (txt,))
            msg = {'text': txt, 'lang': 'en'}
            socket.send(pickle.dumps(msg))
            message = pickle.loads(socket.recv())
            print(str(message))


address = address + str(sys.argv[1])
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(address)
test_annotator(socket)
