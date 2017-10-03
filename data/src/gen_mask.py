#!/bin/env python
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

NS='{http://www.w3.org/2000/svg}'

def round(f):
    return int(10000*f + 0.5)/10000.0

class BBox:
    '''
    BBox - class to hold a bounding-box rectangles.
    '''
    def __init__(self,x,y,w,h):
        assert(w>=0) # fix if w,h are ever negative
        assert(h>=0)
        self.x,self.y,self.w,self.h = x,y,w,h
        self.x1 = self.x + self.w
        self.y1 = self.y + self.h
    def add(self,x,y,w,h):
        assert(w>=0)
        assert(h>=0)
        x1 = x+w
        y1 = y+h
        self.x, self.x1 = min(self.x,x), max(self.x1,x1)
        self.y, self.y1 = min(self.y,y), max(self.y1,y1)
        self.w = self.x1 - self.x
        self.h = self.y1 - self.y
        self.w = round(self.w)
        self.h = round(self.h)
    def __repr__(self):
        return "BBox x0:%f, y0:%f, x1:%f, y1:%f"%(self.x,self.y,self.x1,self.y1)
    def __str__(self):
        return "BBox x0:%f, y0:%f, x1:%f, y1:%f"%(self.x,self.y,self.x1,self.y1)
    def add_svg_rect(self,parent):
        #return '<rect x="%f" y="%f" width="%f" height="%f" fill="pink" />'%(self.x,self.y,self.w,self.h)
        child = ET.SubElement(parent, 'rect')
        child.set('x',str(self.x))
        child.set('y',str(self.y))
        child.set('width',str(self.w))
        child.set('height',str(self.h))
        child.set('fill','pink')
        # modifies parent tree

def xy_from_transform(s):
    '''convert the text "translate(14.2264, 10.8953)" to 14.2264, 10.8953
    and return it.'''
    r,l = s.split(',')
    rr = r.split('(')
    x = float(rr[-1])
    ll = l.split(')')
    y = float(ll[0])
    return x,y

def find_bbox_borders(root):
    # let's find borders of lines
    cur_staff = 0
    cur_staff_lines = 0
    staff_bbs = [] # bounding boxes
    for neighbor in root.iter(NS+'line'):
        #print(neighbor.attrib)
        x0,y0 = xy_from_transform(neighbor.attrib['transform'])
        x1,y1 = float(neighbor.attrib['x1']), float(neighbor.attrib['y1'])
        x2,y2 = float(neighbor.attrib['x2']), float(neighbor.attrib['y2'])
        if cur_staff_lines == 0:
            staff_bbs.append(BBox(x0+x1,y0+y1,x2-x1,y2-y1))
        else:
            staff_bbs[-1].add(x0+x1,y0+y1,x2-x1,y2-y1)
        cur_staff_lines += 1
        if cur_staff_lines == 5:
            cur_staff_lines = 0
            cur_staff += 1
    cur_staff, cur_staff_lines
    return staff_bbs

def prettify_svg(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def print_svg(bbs):
    svg = ET.Element('svg')
    # FIXME - get from original SVG
    svg.set("xmlns","http://www.w3.org/2000/svg")
    svg.set("xmlns:xlink", "http://www.w3.org/1999/xlink")
    svg.set("version", "1.2")
    svg.set("width", "210.00mm")
    svg.set("height", "297.00mm")
    svg.set("viewBox", "0 0 119.5016 169.0094")
    for bb in bbs:
        bb.add_svg_rect(svg)
    print(prettify_svg(svg))

def gen_mask(root):
    bboxes = find_bbox_borders(root)
    print_svg(bboxes)

def main(input_file_name):
    tree = ET.parse(input_file_name)
    root = tree.getroot()
    gen_mask(root)

if __name__ == "__main__":
    main(sys.argv[1])
