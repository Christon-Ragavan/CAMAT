import os

import music21 as m
import music21 as m21



us = m21.environment.UserSettings()
us_path = us.getSettingsPath()
if not os.path.exists(us_path):
    us.create()
us['musescoreDirectPNGPath'] = r'/Applications/MuseScore 3.app/Contents/MacOS/mscore'
us['musicxmlPath'] = r'/Applications/MuseScore 3.app/Contents/MacOS/mscore'
path = "/Users/chris/DocumentLocal/workspace/hfm/weimar/example_files/sacred_xml.xml"
path = "/Users/chris/DocumentLocal/workspace/hfm/weimar/example_files/ultimate_tie_test3.xml"

song = m.converter.parse(path)
# process the ties
song = song.stripTies()

# unfold repetitions
i = 0;
for a in song:
    if a.isStream:
        e = m.repeat.Expander(a)
        s2 = e.process()
        timing = s2.secondsMap
        song[i] = s2
    i += 1;

# todo: add note onsets

def getMusicProperties(x):
    s = '';
    t='';
    s = str(x.pitch) + ", " + str(x.duration.type) + ", " + str(x.duration.quarterLength);
    s += ", "
    if x.tie != None:
        t = x.tie.type;
    s += t + ", " + str(x.pitch.ps) + ", " + str(x.octave); # + str(x.seconds)  # x.seconds not always there
    return s


print('pitch, duration_string, duration, tie, midi pitch, octave')
for a in song.recurse().notes:

    if (a.isNote):
        x = a;
        s = getMusicProperties(x);
        print(s);

    if (a.isChord):
        for x in a._notes:
            s = getMusicProperties(x);
            print(s);

print("Done.")