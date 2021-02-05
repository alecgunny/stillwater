from .data_generator import DataGenerator
from .dummy_data_generator import DummyDataGenerator

# gwpy should not be a hard dependency of this
# library, so wrapping this here. Would be nice
# to be able to catch the attempted import of
# this object and raise a specific error mentioning
# how the user needs to install the necessary libs
# in reality, this should be it's own library that
# imports stillwater, but I need it for now and that
# sounds like a lot of work
try:
    from .low_latency_frame_generator import LowLatencyFrameGenerator
except ImportError as e:
    if "gwpy" not in str(e):
        raise e
