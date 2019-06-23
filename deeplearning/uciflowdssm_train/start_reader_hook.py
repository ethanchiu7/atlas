import tensorflow as tf
from tensorflow.python.training import session_run_hook


class StartStandardKvReaderHook(session_run_hook.SessionRunHook):
    def __init__(self, kv_reader):
        super(StartStandardKvReaderHook, self).__init__()
        print("*" * 50)
        print("create StartStandardKvReaderHook")
        self._kv_reader = kv_reader
        self._is_started = False

    def after_create_session(self, session, coord):
        print('*' * 100)
        print('After creating session.')
        print('*' * 100)
        self._kv_reader.set_session(session)
        self._kv_reader.start()
        self._is_started = True
        pass

    def end(self, session):
        print("end StartStandardKvReaderHook")
        self._is_started = False
        self._kv_reader.stop()

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        pass

