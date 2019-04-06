"""
Extends the Event class to describe a musical rest.

@author Benjie Genchel, based on code by Alexander Dodd
"""

# package uk.ac.kent.computing.gttm.Elements;
# import java.io.Serializable;
from event import Event
from enums import DurationEnum

# public class RestEvent implements Event{
class RestEvent(Event):
    # private DurationEnum length;
    
    # public RestEvent(DurationEnum length)
    def __init__(self, length):
        """
        :param length: (DurationEnum) the duration describing how long the rest should last
        """
        # this.length = length;
        self._length = length

    # @Override
    # public DurationEnum getDurationEnum() {
    def getDurationEnum(self):
        # return length; 
        return self._length
    # }
# }
