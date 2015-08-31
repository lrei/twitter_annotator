#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import zmq
import cPickle as pickle


address = 'tcp://localhost:'

senti_messages_en = ['i hate everything because it sucks :(', 
                     'i love my iphone because apple is the best :)',
                     'the reporter was completely impartial as am i']

senti_messages_es = ['odio vacas tontas estúpidas',
                     'perros y gatos me hacen feliz me gustan :-)',
                     'me lavo las manos después']

def test_annotator(socket):
    while True:
        for txt in senti_messages_en:
            print('Sending Request: %s' % (txt,))
            msg = {'text': txt, 'lang': 'en'}
            socket.send(pickle.dumps(msg))
            message = pickle.loads(socket.recv())
            print(str(message))
        for txt in senti_messages_es:
            print('Sending Request: %s' % (txt,))
            msg = {'text': txt, 'lang': 'es'}
            socket.send(pickle.dumps(msg))
            message = pickle.loads(socket.recv())
            print(str(message))
        break


address = address + str(sys.argv[1])
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(address)
test_annotator(socket)
