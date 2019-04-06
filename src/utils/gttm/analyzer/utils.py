"""
based on alexanderbdodd/GTTM_Music_Generator
"""
# import uk.ac.kent.computing.gttm.Elements.AttackEvent
# import uk.ac.kent.computing.gttm.Elements.DurationEnum
# import uk.ac.kent.computing.gttm.Elements.Event
# import uk.ac.kent.computing.gttm.Elements.Key
# import uk.ac.kent.computing.gttm.Grammar_Elements.ExceptionClasses.GroupingWellFormednessException
# import uk.ac.kent.computing.gttm.Grammar_Elements.GroupingStructure.BaseGroup
# import uk.ac.kent.computing.gttm.Grammar_Elements.GroupingStructure.Group
# import uk.ac.kent.computing.gttm.Grammar_Elements.GroupingStructure.HighLevelGroup
# import uk.ac.kent.computing.gttm.Grammar_Elements.MetricalStructure.Beat
# import uk.ac.kent.computing.gttm.Grammar_Elements.ReductionBranches.Branch

from collections import deque

def getDurationBeatExpansion(duration):
    """
    :param duration: (DurationEnum)
    :return: (int)
    """
    spanDistance = 0

    switch_dict = {
        SIXTEENTH: 0,
        EIGHTH: 1,
        QUARTER: 2,
        HALF: 3,
        WHOLE: 4
    }

    spanDistance = switch_dict[duration]
    return spanDistance

def getAllBoundaryBeats(gGroup, pitchBeats):
    """
    Get all the boundary beats within the whole grammatical structure
    :param gGroup: (HighLevelGroup)
    :param pitchBeats: (List<Beat>)
    :return: (List<Beat>)
    """
    baseGroups = [] # (List<Group>)
    groups = deque() # (List<HighLevelGroup>)

    groups.append(gGroup)
    while not len(groups):
        gr = groups.popleft()
        for subgroup in gr.getSubGroups():
            if subgroup.__class__ == BaseGroup:
                baseGroups.append(group)
            else:
                groups.append(group) # they try to cast this to type HighLevelGroup in original

    return identifyBoundaryBeats(baseGroups, pitchBeats)

def identifyBoundaryBeats(groups, pitchBeats):
    """
    Identifies the beats which constitute boundary pitch events of the given groups and returns them as a list
    :param groups: (List<Group>) the groups to be searched for boundary pitches beats
    :param pitchBeats: (List<Beat>) the list of beats allocated as pitch instantiation beats
    :return: (List<Beat> a list of boundary beats
    """
    beats = [] # (List<Beat>)

    # for all the specified groups
    for group in groups:
        # invoke the identifyBoundaryBeat method for the group
        beat = identifyBoundaryBeat(group, pitchBeats) # (Beat)
        # make sure the beat returned is not None, and add it to the beats list
        if beat is not None:
            beats.append(beat)

    return beats

def identifyBoundaryBeat(group, pitchBeats):
    """
    Used to identify the boundary beat of a group. A boundary beat is the last beat of the group which lands on 
    the inception of a pitch event.
    :param group: (Group) the group to be checked
    :param pitchBeats: (List<Beat>) a list of all beats which are on the inception of a pitch event
    :return: (Beat) the beat identified as the boundary. If no pitch beat is found, then None is returned.
    """
    spanSize = group.getMetricalBeatSpan().size() # (int)
    currentBeat = group.getMetricalBeatSpan().get(spanSize - 1) # (Beat)

    # while the group contains the beat to be assessed
    while group.getMetricalBeatSpan().contains(currentBeat):
        # if the beat is a pitch beat, return the beat
        if pitchBeats.contains(currentBeat):
            return currentBeat
        # reduce the position to search
        spanSize -= 1
        # if the position is now less than or equal to zero,
        # return None to indicate no beat was found
        if spanSize <= 0:
            return None
        # if group.getMetricalBeatSpan().get(spanSize - 1) is not None:
        currentBeat = group.getMetricalBeatSpan().get(spanSize - 1)

    return None


def getBeatSpanFromBoundary(boundaryBeat, group, pitchBeats):
    """
    This is used to identify the span of beats that surround a boundary beat.
    The boundary beat need not necessarily be an actual boundary beat.
    :param boundaryBeat: (Beat) the beat to be used in constructing the beat span
    :param group: (Group) the greatest group to constrain the search for the beats
    :param pitchBeats: (List<Beat>) the list of beats allocated as pitch instantiation beats within
    :return: (List<Beat>) the span of beats
    """
    # Collections.sort(pitchBeats, boundaryBeat)
    sorted(pitchBeats, key=lambda pb: pb.getPosition())
    
    beats = [] # (List<Beat>)
    currentBeat = boundaryBeat # (Beat)

    # find the index of the boundary beat
    beatPos = group.getMetricalBeatSpan().indexOf(currentBeat) (int)
    beatPos -= 1

    # finding the first pitch beat of the span  while the beat position 
    # is within the group's span of beats
    found = False # (bool)
    while (beatPos >= 0) and (beatPos < len(group.getMetricalBeatSpan())):
        currentBeat = group.getMetricalBeatSpan().get(beatPos)
        # check whether it constitutes a pitch beat
        # if so, break out of the while loop
        if currentBeat in pitchBeats:
            found = True
            break
        else:
            beatPos -= 1

    # if no first beat was found, return None for the beat span list
    # else begin searching for the next pitch beat to be refactored
    if not found:
        return None
    else:
        beatPos = 1 # else begin searching for the next pitch beat to be refactored
        beats.append(currentBeat)

        # loop until 4 pitch beats have been identified
        while beatPos < 4:
            if currentBeat.getNextBeat() is None:
                # if the next beat is None, then the constraint of
                # finding four pitch beats can't be met, so return None
                return None

            currentBeat = currentBeat.getNextBeat()
            # if it's a pitch beat, increase the total of found pitch beats
            if pitchBeats.contains(currentBeat):
                beatPos += 1

            # add the beat to the span
            beats.append(currentBeat)

    # adding all the beats after the final pitch beat which
    # occur before the next inception of a pitch beat
    while (currentBeat.getNextBeat() is not None) and (currentBeat.getNextBeat() not in pitchBeats):
        currentBeat = currentBeat.getNextBeat()
        beats.append(currentBeat)

    # sort of the beats so that they are in position order
    sorted(beats, lambda b: b.getPosition())
    return beats

def getBeatSpanFromBeat(b, pitchBeats):
    """
    :param b: (Beat)
    :param pitchBeats: (List<Beat>)
    :return: (List<Beat>)
    """
    if b not in pitchBeats:
        return None

    if b.getPreviousBeat() is None:
        return None

    firstSpanBeat = b.getPreviousBeat()
    while firstSpanBeat not in pitchBeats:
        if firstSpanBeat.getPreviousBeat() is not None:
            firstSpanBeat = firstSpanBeat.getPreviousBeat()
        else:
            return None

    beatSpan = [] # (List<Beat>)
    beatSpan.append(firstSpanBeat)

    pitchCount = 1 # (int)
    while pitchCount < 4:
        if firstSpanBeat.getNextBeat() is not None:
            firstSpanBeat = firstSpanBeat.getNextBeat()
            pitchCount += (firstSpanBeat in pitchBeats)
            beatSpan.append(firstSpanBeat)
        else:
            return None
    
    if firstSpanBeat.getNextBeat() is not None:
        firstSpanBeat = firstSpanBeat.getNextBeat()
        while firstSpanBeat not in pitchBeats:
            beatSpan.append(firstSpanBeat)
            if firstSpanBeat.getNextBeat() is not None:
                firstSpanBeat = firstSpanBeat.getNextBeat()
            else:
                break

    sorted(beatSpan, key=lambda bs: bs.getPosition())
    return beatSpan

def createPitchEventList(chain, beatSpan):
    """
    :param chain: (Map<Beat, Key>)
    :param beatSpan: (List<Beat>)
    :return: (List<Event>)
    """
    List<Event> events = []

    for beat in beatSpan:
        if beat in chain.keys():
            events.append(AttackEvent(chain[beat], EIGHTH))

    return events
    
def getEventStreamFragment(beatMap, startBeat, endBeat):
    """
    :param beatMap: (Map<Beat, Event>)
    :param startBeat: (Beat)
    :param endBeat: (Beat)
    :return: (List<Event>)
    """
    events = [] # (List<Event>)
    while startBeat != endBeat:
        if startBeat in beatMap.keys():
            events.append(beatMap[startBeat])

        startBeat = startBeat.getNextBeat()

    if endBeat in beatMap.keys():
        events.append(beatMap[endBeat])
         
    return events
 
def calculateDistanceFromBranch(topBranch, childBranch):
    """
    Calculates how many branches must be traversed from the top branch to get to
    the given child branch. The child branch must be on a lower level than the top branch.
    :param topBranch: (Branch)
    :param childBranch: (Branch)
    :return: (int)
    """
    if topBranch == childBranch:
        return 0
    
    distance = 1 # (int)
    return calculateDistanceFromBranch(topBranch, childBranch, 1)
    

def calculateDistanceFromBranch(topBranch, childBranch, distance):
    """
    :param topBranch: (Branch)
    :param childBranch: (Branch)
    :param distance: (int)
    :return: (int)
    """
    if topBranch.getChildBranches()[childBranch.getLevel()] == childBranch:
        return distance

    distance += 1

    for level in topBranch.getOrderedLevels(): # level is type Integer
        # solution = -1 # (int)
        solution = calculateDistanceFromBranch(topBranch.getChildBranches()[level], childBranch, distance)
        if solution != -1:
            return solution

    return -1

    
def getBoundaryEventsPerLevel(pitchBeats, gGroup):
    """
    :param pitchBeats: (List<Beat>)
    :param gGroup: (HighLevelGroup)
    :return: (Map<Integer, List<Group>>)
    """
    l2g_map = {} # (Map<Integer, List<Group>>) level to group
    boundaryBeats = [] # (List<Beat>)
    nextLevelGroup = gGroup.getSubGroups() # (List<Group>)

    level = 1 # (Integer)
    while len(nextLevelGroup):
        boundaryBeats.extend(identifyBoundaryBeats(nextLevelGroup, pitchBeats))
        l2g_map[level] = nextLevelGroup

        tempGroup = None # (HighLevelGroup)
        if len(nextLevelGroup) > 1:
            try:
                tempGroup = HighLevelGroup(nextLevelGroup)
            except GroupingWellFormednessException as e:
                print("{}: {}".format(type(e), str(e)))
        elif nextLevelGroup[0].__class__ == HighLevelGroup:
            tempGroup = nextLevelGroup[0] # cast to HighLevelGroup

        if tempGroup is not None:
            nextLevelGroup = getNextLevelGroups(tempGroup, boundaryBeats, pitchBeats)
            level += 1
        else:
            break

    return l2g_map


#######
# Definitely going to have to fix up whatever this is
######
def getNextLowestDuration(duration):
    """
    Return the next lowest duration.
    :param duration: (DurationEnum)
    :return: (DurationEnum)
    """
    switch_map = {
            WHOLE: DurationEnum.HALF,
            HALF: DurationEnum.QUARTER,
            QUARTER: DurationEnum.EIGHTH,
            SIXTEENTH: None}

    return switch_map[duration]


def createMockHighLevelGroup(groups, pitchBeats):
    """
    Creates a highlevel groups containing all of the subgroups
    of all the groups in the groups list.
    :param groups: (List<Group>)
    :param pitchBeats: (List<Beat>)
    :return: (List<Group>)
    """
    subGroups = [] # (List<Group>)
    # for all the groups
    for gr in groups:
        # if the group is a highlevel group
        # add all the sublevel groups of the group to the list of subgroups
        if gr.__class__ == HighLevelGroup:
            subGroups.extend(hg.getSubGroups())

    return subGroups


def getNextLevelGroups(group, boundaryBeats, pitchBeats):
    """
    This goes through all the subgroups of the given HighLevelGroup and for each group,
    provided it doesn't have a boundary beat contained within the boundary beats list,
    adds it to the list of groups which constitute the next level of groups in the hierarchy.
    :param group: (HighLevelGroup)
    :param boundaryBeats: (List<Beat>)
    :param pitchBeats: (List<Beat>)
    :return: (List<Group>)
    """
    nextLevel = [] # (List<Group>)
    # get all the sub-sub groups of the group and add them to the nextLevel beats
    # groups which have already seen boundary beats are not added.        
    for g in createMockHighLevelGroup(group.getSubGroups(), pitchBeats):
        if identifyBoundaryBeat(g, pitchBeats) not in boundaryBeats:
            nextLevel.add(group2)
    return nextLevel


######
# This is not necessary for python, probably will delete it later
######
def listToArray(dlist):
    """
    Converts a list of Double objects to an array of double values.
    :param dlist: (List<Double>) the list of Double objects
    :return: (double[]) the array of double values.
    """
    array = [0.0] * len(dlist) # (double[])
    for i, num in enumerate(dlist):
        array[i] = num

    return array
