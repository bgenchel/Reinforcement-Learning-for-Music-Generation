"""
A class extending the AttackEvent class to represent the instantiation of multiple pitches

@author Benjie Genchel, adapted from code by Alexander Dodd
ref: https://github.com/alexanderbdodd/GTTM_Music_Generator
"""

# package uk.ac.kent.computing.gttm.Elements;
from attack_event import AttackEvent

# public class AttackEventChord extends AttackEvent {
class AttackEventChord(AttackEvent):
    # private Chord chrd;

    # public AttackEventChord(Chord chrd, DurationEnum length) {
    def __init__(self, chrd, length):
        # super(chrd.getKeys().get(0), length);
        super().initFromKey(chrd.getKeys()[0], length)
        # this.chrd = chrd;
        self.chrd = chrd
    # }

    # public Chord getChord() {
    def getChord(self):
        """
        :return: (Chord) the pitches associated with this AttackEvent
        """
        # return chrd;
        return self.chrd
    # }
# }
