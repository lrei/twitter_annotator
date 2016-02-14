# -*- coding: utf-8 -*-
"""
Luis Rei <luis.rei@ijs.si> @lmrei http://luisrei.com
version 1.1
10 Feb 2016
To terminate:

    kill -s INT <pid>

ZMQ Load Balancer/Worker based on example by
Brandon Carpenter (hashstat) <brandon(dot)carpenter(at)pnnl(dot)gov>
http://zguide.zeromq.org/py:lbbroker
"""

import logging
import multiprocessing
import threading
import json
import zmq
from zmq.eventloop import ioloop, zmqstream
from functools import partial

ioloop.install()

import tornado
from tornado import web


def worker_task_builder(worker_f, backend_address):
    """Returns the multiprocess worker the function that calls
    process_message()
    """

    def worker_task(worker_id):
        # setup service
        socket = zmq.Context().socket(zmq.REQ)
        socket.identity = u"Worker-{}".format(worker_id).encode("ascii")
        socket.connect(backend_address)

        # signal to the broker that we are ready
        socket.send(b'READY')

        # start working (pun intended)
        while True:
            address, _, msg = socket.recv_multipart()
            reply = {'error': 'none'}

            try:
                data = json.loads(msg)
                reply = worker_f(data=data)
            except Exception as e:
                logging.exception(e)
                reply = {'error': str(e)}

            reply = json.dumps(reply)
            socket.send_multipart([address, b"", reply])

    return worker_task


def zserve(worker_task, n_workers, backend_address, frontend_address):
    """Load balancer: Starts the workers (in different processes) and
    balances the work it receives from a client between the different worker
    processes.
    """

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

    # Clean up
    backend.close()
    frontend.close()
    context.term()


class WebHandler(tornado.web.RequestHandler):
    def initialize(self, address):
        self.address = address

    @web.asynchronous
    def get(self):
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        s.connect(self.address)

        # get the parameters
        try:
            lang = self.get_query_argument('lang')
            text = self.get_query_argument('text')
            jsdata = json.dumps({'text': text, 'lang': lang})

            # send request to worker
            s.send(jsdata)
            self.stream = zmqstream.ZMQStream(s)
            self.stream.on_recv(self.handle_reply)
        except Exception as ex:
            self.write({'error': str(ex)})
            self.finish()

    def handle_reply(self, msg):
        # finish web request with worker's reply
        reply = json.loads(msg[0])
        self.stream.close()
        self.write(reply)
        self.finish()


def serve(port, worker_task, n_workers, backend_address, frontend_address):

    zserver_f = partial(zserve, worker_task, n_workers, backend_address,
                        frontend_address)
    worker = threading.Thread(target=zserver_f)
    worker.daemon = True
    worker.start()

    d = {'address': frontend_address}

    import sys

    def dot():
        """callback for showing that IOLoop is still responsive while we wait
        """
        sys.stdout.write('.')
        sys.stdout.flush()

    application = tornado.web.Application([(r"/", WebHandler, d)])
    beat = ioloop.PeriodicCallback(dot, 1000)
    beat.start()
    application.listen(port)
    try:
        ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        print('Interrupted')
