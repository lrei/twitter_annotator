#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# xLiMe Twitter Annotator
Luis Rei <luis.rei@ijs.si> @lmrei
26 Aug 2015


To terminate:

    kill -s INT <pid>


ZMQ Load Balancer/Worker based on example by
Brandon Carpenter (hashstat) <brandon(dot)carpenter(at)pnnl(dot)gov>
http://zguide.zeromq.org/py:lbbroker
"""

import os
import argparse
import zmq
import multiprocessing
import cPickle as pickle

import settings
from gracefulinterrupthandler import GracefulInterruptHandler


IDENTIFIER = 'jsi_xlime_'
BACKEND_ADDRESS = 'ipc://annotbackend.ipc'
DEFAULT_PORT = 1984
DEFAULT_NUM_WORKERS = 4


def process_message(router, data):
    """This is the function that actually processes the data
        0 - Language Routing
        1 - Sentiment
        2 - TODO
    """
    reply = data

    # message must have a lang attribute
    if 'lang' not in data:
        return reply

    lang = data['lang']

    # check if we are setup to handle this language
    if lang not in router:
        return reply

    # check if message has a text property
    if 'text' not in data:
        return reply

    text = data['text'].strip()

    # check that text field is not empty
    if not text:
        return reply

    #
    # Pipeline begins
    #
    tokenizer = router[lang]['tokenizer']
    text = tokenizer(text)
    tokens = text.split()
    property = IDENTIFIER + 'tokenized'
    reply[property] = text

    #
    # 1 - Sentiment
    #
    property = IDENTIFIER + 'sentiment'
    classifier = router[lang]['sentiment']
    model = router[lang]['sentiment_model']

    # Preprocess text for Sentiment
    preprocessor = router[lang]['preprocessor']
    text_pp = preprocessor(text)

    # Sentiment classification
    reply[property] = classifier(model, text_pp)

    #
    # 2 - PoS
    #
    property = IDENTIFIER + 'pos'

    pos_model = router[lang]['pos_model']
    pos_tag = router[lang]['pos']

    reply[property] = pos_tag(pos_model, tokens)

    #
    # 3 - NER
    #
    property = IDENTIFIER + 'ne'

    ner_model = router[lang]['ner_model']
    ner = router[lang]['ner']

    reply[property] = ner(ner_model, tokens)


    # finally return:
    return reply


def worker_task(worker_id):
    """The multiprocess worker - the function that calls process_message()
    """
    # setup router
    router = settings.router
    # setup service
    socket = zmq.Context().socket(zmq.REQ)
    socket.identity = u"Worker-{}".format(worker_id).encode("ascii")
    socket.connect(BACKEND_ADDRESS)

    # signal to the broker that we are ready
    socket.send(b'READY')

    # start working (pun intended)
    while True:
        address, _, msg = socket.recv_multipart()
        err_property = IDENTIFIER + 'error'
        reply = {err_property: 'none'}

        try:
            data = pickle.loads(msg)
            reply = process_message(router, data)
        except Exception as e:
            reply = {err_property: str(e)}

        reply = pickle.dumps(reply)
        socket.send_multipart([address, b"", reply])


def load_balancer(port=DEFAULT_PORT, n_workers=DEFAULT_NUM_WORKERS):
    """Load balancer: Starts the workers (in different processes) and
    balances the work it receives from a client between the different worker
    processes.
    """
    frontend_address = 'tcp://*:' + str(port)
    backend_address = BACKEND_ADDRESS

    # Prepare context and sockets
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(frontend_address)
    backend = context.socket(zmq.ROUTER)
    backend.bind(backend_address)

    # Start Workers
    def start(task, *args):
        """A function that starts background processes"""
        process = multiprocessing.Process(target=task, args=args)
        process.daemon = True
        process.start()

    for i in range(n_workers):
        start(worker_task, i)

    # Initialize main loop state
    workers = []
    poller = zmq.Poller()

    poller.register(backend, zmq.POLLIN)

    with GracefulInterruptHandler() as h:
        while True:
            sockets = dict(poller.poll())

            #
            # Handle worker activity on the backend
            #
            if backend in sockets:
                request = backend.recv_multipart()
                worker, _, client = request[:3]

                if not workers:
                    # Client polling was suspended - unsuspend it:
                    # (if we got a msg from a worker it means it is availabe
                    # Poll for clients now that a worker is available
                    poller.register(frontend, zmq.POLLIN)

                workers.append(worker)
                if client != b"READY" and len(request) > 3:
                    # If client reply, send rest back to frontend
                    _, reply = request[3:]
                    frontend.send_multipart([client, b"", reply])
            #
            # Get next client request, route to last-used worker
            #
            if frontend in sockets:
                client, _, request = frontend.recv_multipart()
                worker = workers.pop(0)
                backend.send_multipart([worker, b"", client, b"", request])
                if not workers:
                    # Suspend client polling
                    # Don't poll clients if no workers are available
                    poller.unregister(frontend)

            if h.interrupted:
                break

    # Clean up
    backend.close()
    frontend.close()
    context.term()


def main():
    parser = argparse.ArgumentParser(description='Run SGD.')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help='read/write to zmq socket at specified port')

    # common options
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_NUM_WORKERS,
                        help='number of cores to use in parallel')
    # Parse
    args = parser.parse_args()

    # Print PID
    print(str(os.getpid()))

    # Run forever (or until kill -INT)
    load_balancer(port=args.port, n_workers=args.n_jobs)

    print('quit')


if __name__ == '__main__':
    main()
