"""
A class used to describe instantiations of a pitch event

@author Benjie Genchel, adapted from code by Alexander Dodd
ref: https://github.com/alexanderbdodd/GTTM_Music_Generator
"""

# package uk.ac.kent.computing.gttm.Elements;
from key import Key
from event import Event
from enums import *

# public class AttackEvent extends Key implements Event {
class AttackEvent(Key, Event):
    # private DurationEnum duration;
    # private DynamicsEnum dynamicValue;
    # private ArticulationEnum articulationValue = ArticulationEnum.NONE;
    articulationValue = ArticulationEnum.NONE

    # public AttackEvent(KeyLetterEnum letter, KeyPositionEnum position, DurationEnum length) {
    def __init__(self, letter, position, length):
        """
        Creates an AttackEvent object.
        :param letter: (KeyLetterEnum) the key of the pitch
        :param position: (KeyPositionEnum) the position of the pitch
        :param length: (DurationEnum) the duration of the pitch
        """
        # super(letter, position);
        super().__init__(letter, position)
        # duration = length;
        this._duration = length
        # dynamicValue = DynamicsEnum.MP;
        this._dynamicValue = DynamicsEnum.MP
    # }

    # public AttackEvent(Key key, DurationEnum length) {
    @classmethod
    def initFromKey(cls, key, length):
        """
        Create an AttackEvent from a Key object
        :param key: the pitch of the AttackEvent
        :param length: the duration of the AttackEvent
        """
        # super(key);
        # duration = length;
        # dynamicValue = DynamicsEnum.MP;
        return cls(key.getNote(), key.getPosition(), length)
    # }

    # public AttackEvent(KeyLetterEnum letter, KeyPositionEnum position, DurationEnum length, DynamicsEnum dynamic) {
    @classmethod
    def initWithDynamic(cls, letter, position, length, dynamic):
        """
        Creates an AttackEvent with a dynamic
        :param letter: (KeyLetterEnum) the key of the pitch
        :param position: (KeyPositionEnum) the position of the pitch
        :param length: (DurationEnum) the duration of the pitch
        :param dynamic: (DynamicsEnum) the dynamic value to associate with the event
        """
        # super(letter, position);
        # duration = length;
        # dynamicValue = dynamic;
        return cls(letter, position, length, dynamic)
    # }

    # public AttackEvent(Key key, DurationEnum length, DynamicsEnum dynamic) {
    @classmethod
    def initFromKeyWithDynamic(cls, key, length, dynamic):
        """
        Create an AttackEvent with a dynamic value.
        :param key: (Key) the pitch of the AttackEvent
        :param length: (DurationEnum) the duration of the AttackEvent
        :param dynamic: (DynamicsEnum) the dynamic value to associate with the event
        """
        # super(key);
        # duration = length;
        # dynamicValue = dynamic;
        return cls(key.getNote(), key.getPosition(), length, dynamic)
    # }

    # public DurationEnum getDurationEnum() {
    def getDurationEnum(self):
        """
        return the DurationEnum associated with this AttackEvent 
        :return: (DurationEnum)
        """
        return self._duration

    # public DynamicsEnum getDynamic() {
    def getDynamic(self):
        """
        return the DynamicsEnum associated with this AttackEvent 
        :return: (DynamicsEnum)
        """
        # return dynamicValue;
        return self._dynamicValue
    # }

    # public void setDynamic(DynamicsEnum dynamic) {
    def setDynamic(self, dynamic):
        """
        Set a new dynamic value for this AttackEvent
        :param dynamic: (DynamicsEnum) the dynamic value to give the AttackEvent
        """
        # dynamicValue = dynamic;
        self._dynamicValue = dynamic
    # }

    # public void setArticulationValue(ArticulationEnum articulationValue) {
    def setArticulationValue(articulationValue):
        """
        Set an articulation value to associate with this AttackEvent
        :param articulationValue: the ArticulationEnum to associate with this AttackEvent 
        """
        this._articulationValue = articulationValue;
    # }

    # public ArticulationEnum getArticulationValue() {
    def getArticulationValue(self):
        """
        :return: (ArticulationEnum) the ArticulationEnum associated with this AttackEvent
        """
        # return articulationValue;
        return self._articulationValue
    # }
# }
