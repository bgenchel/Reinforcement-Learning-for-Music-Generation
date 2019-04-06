"""
Provides an interface for Event classes. In GTTM, Events encapsulate the instantiation of
various musical elements, such as rests and pitches.

@author Benjie Genchel, adapted from code by Alexander Dodd
ref: https://github.com/alexanderbdodd/GTTM_Music_Generator
"""
# package uk.ac.kent.computing.gttm.Elements;
import abc

# public interface Event {
class Event(abc.ABC):

    def __init__(self, *args, **kwargs):
        pass

    # public DurationEnum getDurationEnum();
    @abc.abstractmethod
    def getDurationEnum(self):
        """
        :return: the duration associated with the Event
        """
        pass
# }
