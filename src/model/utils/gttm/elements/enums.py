"""
GTTM Analyzer Enums 

@author Benjie Genchel, adapted from code by Alexander Dodd 
ref: https://github.com/alexanderbdodd/GTTM_Music_Generator
"""

class pyEnum: 
    class Member:
        def __init__(self, name):
            self._name = name

        def __str__(self):
            return self._name

    member_dict = None
    members = []

    @classmethod
    def __getattr__(cls, name):
        if cls.member_dict is None:
            cls.member_dict = {m._name: m for m in cls.members}

        if name in cls.member_keys:
            return name
        raise AttributeError

    @classmethod
    def values(cls):
        return cls.members 


class ArticulationEnum(pyEnum):
    """
    Used to describe different musical articulations
    """
    members = [
        Member("None"),
        Member("STACCATO"),
        Member("SLUR"),
        Member("SLURALT"),
    ]


class DurationEnum(pyEnum):
    """
    Used to describe the length of a pitch event.
    """
    class Member(pyEnum.Member):
        def __init__(self, name, length):
            super().__init__(name)
            self._length = length

        def getLength(self):
            return self._length

    members = [
        Member("SIXTEENTH", 1),
        Member("EIGHTH", 2),
        Member("QUARTER", 4),
        Member("HALF", 8),
        Member("WHOLE", 16)
    ]


class DynamicsEnum(pyEnum):
    """
    A list of dynamics that can be associated with an AttackEvent or a group of music.
    """
    members = [
        Member("PP"), Member("P"), Member("MP"), Member("MF"), Member("F"), Member("FF")
    ]

package uk.ac.kent.computing.gttm.Elements;

import java.io.Serializable;
import java.util.*;

class KeyLetterEnum(pyEnum):
    """
    The KeyLetterEnum is used to describe key positions within a scale for a
    pitch, and is used in conjunction with the KeyPositionEnum to describe a
    pitch position.
    """
    class Member(pyEnum.Member):
        def __init__(self, name, keyNumber, keyIdentity):
            super().__init__(name)
            self.keyNumber = keyNumber
            self.keyIdentity = keyIdentity

        def getKeyNumber(self):
            """
            Returns a number that is shared by all keys in the same position. E.g. b sharp and c have the same key
            number.
            """
            return self.keyNumber

        def getKeyIdentity(self):
            """
            Returns a the key letter's unique number
            """
            return self.keyIdentity

    members = [
        Member("C", 0, 0), Member("CS", 1, 1), Member("Df", 1, 2), Member("D", 2, 3), Member("DS", 3, 4), 
        Member("Ef", 3, 5), Member("E", 4, 6), Member("Ff", 4, 7), Member("ES", 5, 8), Member("F", 5, 9), 
        Member("FS", 6, 10), Member("Gf", 6, 11), Member("G", 7, 12), Member("GS", 8, 13), Member("Af", 8, 14),
        Member("A", 9, 15), Member("AS", 10, 16), Member("Bf", 10, 17), Member("B", 11, 18), Member("Cf", 11, 19),
        Member("BS", 0, 20)
    ]

    # public static int[] getUniquePositions() {
    @classmethod
    def getUniquePositions(cls):
        """
        return an array of the unique key positions
        """
        # int[] positions = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        # return positions;
        return list(set([v.getKeyNumber() for v in cls.values()]))

    # public static KeyLetterEnum[] getNextNote(KeyLetterEnum current) {
    @classmethod
    def getNextNote(cls, current) {
        """
        Returns an array of the next key letters after the given key letter.
        :param current: (KeyLetterEnum) the key letter from which to assess the next note
        :return: (KeyLetterEnum) an array of the next key letters enums directly after the given Key letter
        """
        # KeyLetterEnum[] array = new KeyLetterEnum[2];
        array = []
        # int nextNumber;
        # if (current.getKeyNumber() + 1 < 12) {
        #     nextNumber = current.getKeyNumber() + 1;
        # } else {
        #     nextNumber = 0;
        # }
        nextNumber = 0
        if (current.getKeyNumber() + 1) < 12:
            nextNumber = current.getKeyNumber() + 1

        # int position = 0;
        position = 0
        # boolean pastNext = false;
        pastNext = False
        # int arrayPos = 0;
        arrayPos = 0

        # KeyLetterEnum[] keysArray = GeneralManipulator.rearrangeNotes(current, KeyLetterEnum.values()).toArray(new KeyLetterEnum[0]);
        keysArray = GeneralManipulator.rearrangeNotes(current, cls.values())

        while not pastNext:
            if keysArray[position].getKeyNumber() == nextNumber:
                array[arrayPos] = keysArray[position]
                arrayPos += 1
                position += 1
                continue

            if keysArray[position].getKeyNumber() == current.getKeyNumber():
                position += 1
                continue

            pastNext = True
            position += 1

        return array


    # public static KeyLetterEnum[] getPreviousNote(KeyLetterEnum current) {
    @classmethod
    def getPreviousNote(cls, current):
        """
        Returns an array of the last key letters after the given key letter.
        :param current: (KeyLetterEnum) the key letter from which to assess the previous notes
        :return: (KeyLetterEnum[]) an array of the next key letters enums directly before the given Key letter
        """
        # KeyLetterEnum[] array = new KeyLetterEnum[2];
        array = []

        # int nextNumber;
        # if (current.getKeyNumber() - 1 >= 0) {
        #     nextNumber = current.getKeyNumber() - 1;
        # } else {
        #     nextNumber = 11;
        # }
        nextNumber = 11
        if (current.getKeyNumber() - 1) >= 0:
            nextNumber = current.getKeyNumber() - 1

        # int position = 0;
        # boolean pastNext = false;
        # int arrayPos = 0;
        position = 0
        pastNext = False
        arrayPos = 0

        # List<KeyLetterEnum> keysArray = GeneralManipulator.rearrangeNotes(current, KeyLetterEnum.values());
        keysArray = GeneralManipulator.rearrangeNotes(current, cls.values())
        # Collections.reverse(keysArray);
        keysArray.reverse() # in place
        # keysArray = GeneralManipulator.rearrangeNotes(current, keysArray.toArray(new KeyLetterEnum[0]));
        keysArray = GeneralManipulator.rearrangeNotes(current, cls.values())

        while not pastNext:
            if keysArray[position].getKeyNumber() == nextNumber:
                array[arrayPos] = keysArray[position]
                arrayPos += 1
                position += 1
                continue;

            if keysArray[position].getKeyNumber() == current.getKeyNumber():
                position += 1
                continue

            pastNext = True
            position += 1

        return array


class KeyPositionEnum(pyEnum)
    """
    Describes the position of a KeyLetterEnum in order to describe a specific pitch position.
    """
    class Member(pyEnum.Member):
        def __init__(self, name, position):
            super().__init__(name)
            self._position = position

        def getPosition(self):
            return self._position

    members = [
        Member("FIRST", 1), Member("SECOND", 2), Member("THIRD", 3), Member("FOURTH", 4), Member("FIFTH", 5),
        Member("SIXTH", 6), Member("SEVENTH", 7), Member("EIGHTH", 8)
    ]

    # public static KeyPositionEnum getNextPosition(KeyPositionEnum currentPosition) {
    @classmethod
    def getNextPosition(cls, currentPosition):
        """
        Returns the position after the given position. If the input position is
        the last position, then the first position is returned.
        :param currentPosition: (KeyPositionEnum) the position from which to find the next position
        :return: (KeyPositionEnum) the position after the current position
        """
        # boolean returning = false;
        # for(KeyPositionEnum keyPos : KeyPositionEnum.values()) {
        #    if(returning)
        #        return keyPos;
        #    if(keyPos == currentPosition)
        #        returning = true;
        # }
        returning = False;
        for keyPos in cls.values():
            if keyPos == currentPosition:
                returning = True
            if returning:
                return True
       
       return None

    # public static KeyPositionEnum getLastPosition(KeyPositionEnum currentPosition) {
    @classmethod
    def getLastPosition(cls, currentPosition):
    """
    Returns the position before the given position. If the input position is
    the last position, then the first position is returned.
    :param currentPosition: (KeyPositionEnum) the position from which to find the last position
    :return: (KeyPositionEnum) the position before the current position
    """
    # KeyPositionEnum last = KeyPositionEnum.values()[KeyPositionEnum.values().length - 1];
    # for(KeyPositionEnum keyPos : KeyPositionEnum.values()) {
    # if(keyPos == currentPosition)
    #     return last;
    # else
    #     last = keyPos;
    # }
       # return KeyPositionEnum.values()[KeyPositionEnum.values().length];
    # }
    last = cls.values()[-1]
    for keyPos in cls.values():
        if keyPos == currentPosition:
            return last
        else:
            last = keyPos

    return clas.values()[0]
