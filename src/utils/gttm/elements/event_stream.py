"""
An EventStream is used to encapsulate a stream of Event objects which is used
to encapsulate a piece of music.
@author Benjie Genchel, adapted from code by Alexander Dodd
"""
# package uk.ac.kent.computing.gttm.Elements;
# import java.util.List;

# public class EventStream {
class EventStream:
    # private List<Event> eventStream;
    # private Scale localScale;

    # public EventStream(List<Event> events, Scale localScale) {
    def __init__(self, events, localScale):
        """
        Create an EventStream object.
        :param events: (List<Event>) a list of Event objects which represents the stream of events
        :param localScale: (Scale) the scale associated with the EventStream
        """
        # eventStream = events;
        self._eventStream = events
        # this.localScale = localScale;
        self._localScale = localScale
    # }

    # public Scale getLocalScale() {
    def getLocalScale(self):
        """
        :return: (Scale) the Scale associated with this EventStream object
        """
        # return localScale;
        return self._localScale
    # }

    # public Event getEvent(int position) {
    def getEvent(self, position):
        """
        Retrieves an Event object at the specified position.
        :param position: (int) the position from which to retrieve the Event object
        :return: (Event) an Event object at the given position, or null if the position is
        greater than the EventStream length
        """
        # if (eventStream.size() <= position) {
        if len(self._eventStream) <= position:
            # return null;
            return None
        # } else {
        else:
            # return eventStream.get(position);
            return self._eventStream[position]
        # }
    # }

    # public List<Event> getEventList() {
    def getEventList(self):
        """
        :return: (List<Event>) an ordered List containing the stream of Event objects
        """
        # return eventStream;
        return self._eventStream
    # }
# }
