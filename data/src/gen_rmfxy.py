#!/usr/bin/env python3
"""
gen_rmf - generate a romer music file
"""
import sys
import xml.etree.ElementTree as ET

class LyNote(object):
    def __init__(self,time,num,dur,pc):
        self.time = float(time)*4
        self.num = int(num)
        self.dur = float(dur)*4
        s,c,r = pc.strip().split()
        self.ly_row, self.ly_col = int(r), int(c)
    def __str__(self):
        return "note,%d,%f,%f"%(self.num,self.time,self.dur)
    def with_point(self,points):
        x,y = points[self.ly_row,self.ly_col]
        return "note,%d,%f,%f,%f,%f"%(self.num,self.time,self.dur,x,y)

def xy_from_transform(ts):
    '''convert the text "translate(14.2264, 10.8953) scale..." to 14.2264, 10.8953
    and return it.'''
    t,s = ts.split(' s')
    r,l = t.split(',')
    rr = r.split('(')
    x = float(rr[-1])
    ll = l.split(')')
    y = float(ll[0])
    return x,y

def parse_lypoints(svg_file_name):
    tree = ET.parse(svg_file_name)
    root = tree.getroot()
    points = {}
    for a in root.findall('{http://www.w3.org/2000/svg}a'):
        fields = a.attrib['{http://www.w3.org/1999/xlink}href'].split(':')
        if fields[0] == 'textedit':
            row,col = int(fields[2]),int(fields[3])
            p = a.find('{http://www.w3.org/2000/svg}path')
            x,y = xy_from_transform(p.attrib['transform'])
            points[(row,col)] = (x,y)
    return points

def parse_line(line):
    fields = line.split('\t')
    if len(fields) > 2:
        if fields[1] == 'note':
            (time,n,note_num,note_len,note_dur,pc) = fields
            return LyNote(time,note_num,note_dur,pc)
    return None

def parse_lynotes(notes_file_name):
    notes = []
    with open(notes_file_name,'r') as note_file:
        for line in note_file:
            note = parse_line(line)
            if note:
                #print(note)
                notes.append(note)
    return notes

def main(notes_file_name,svg_file_name,output_file_base):
    notes = parse_lynotes(notes_file_name)
    points = parse_lypoints(svg_file_name)
    print("generating %s"%(output_file_base+'.rmf'))
    with open(output_file_base+'.rmf','w') as of:
        for n in notes:
            print(n,file=of)
    print("generating %s"%(output_file_base+'.rmfxy'))
    with open(output_file_base+'.rmfxy','w') as of:
        for n in notes:
            print(n.with_point(points),file=of)

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])
