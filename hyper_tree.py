import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random 

''' HyperRectTree

Create an arbitrarily dimensional hyper-rectanglar cell that can be subdivided and searched efficiently

coords - (x0, y0, z0, ..., x1, y1, z1, ...) where (V0, V1) represent opposite vertices of the hyper-rectangle
depth - current recursion depth
max_depth - maximum recursion depth
'''
class HyperRectTree(object):
    def __init__(self, coords, depth=0, max_depth=12):
        self.n = len(coords)//2
        self.coords = coords
        self.depth = depth
        self.max_depth = max_depth
        self.rects = None

    def is_in_cell(self, coord):
        ''' is_in_cell
        
        coord - list, tuple or array of a point in n-dim euclidean space

        returns boolean value
        '''
        bound_all = True
        for i in range(0, self.n):
            bound_i = (coord[i] <= self.coords[self.n + i]) and (coord[i] >= self.coords[i])
            bound_all = bound_all and bound_i
        return bound_all
    
    def get_cell(self, coord):
        ''' get_cell

        coord - list, tuple or array of a point in n-dim euclidean space

        returns rect object
        '''
        if not self.is_in_cell(coord):
            return None
        elif self.rects == None:
            return self
        else:
            for r in self.rects:
                if r.is_in_cell(coord):
                    return r.get_cell(coord)

    def get_coords(self):
        return self.coords

    def unsubdivide(self):
        ''' unsubdivide

        1. Unregister the rects from the tree
        '''
        self.rects = None

    def get_midpoint(self):
        return [(self.coords[self.n + i] + self.coords[i])/2 for i in range(0, self.n)]

    def get_rand_in_cell(self):
        return [np.random.uniform(self.coords[i], self.coords[self.n + i]) for i in range(self.n)]

    def get_even_per_cell(self, m):
        spaces = []
        for i in range(0, self.n):
            spaces.append(np.linspace(self.coords[i], self.coords[self.n + i], m+1, endpoint=False)[1:])

        pts = []
        for pt in itertools.product(*spaces):
            pts.append(pt)

        return pts
        

    def get_all(self):
        if self.rects != None:
            rets = []
            for r in self.rects:
                rets.append(r.get_all())
            return list(itertools.chain(*rets))
        else:
            return self.get_midpoint()

    def subdivide(self):
        ''' subdivide

        1. get midpoint between two opposing vertices
        2. construct a vector whose equal to the longest diagonal
        3. get half of the hyper-rectangle's vertices
            a. augment the first vertex and midpoint into a 2 x n matrix
            b. calculate the cartesian product of all column vectors
        4. The new vertices for each v in vertices is (v, v + vector) 
        '''
        if self.rects == None and self.depth <= self.max_depth:
            rects = []
            mid_v = self.get_midpoint()
            origs = self.coords[:self.n]
            pts_matrix = np.array([origs, mid_v])
            mid_vector = np.array(mid_v) - np.array(origs)
            for i in itertools.product(*pts_matrix.T.tolist()):
                    rects.append(HyperRectTree((list(i) + (np.array(i) + mid_vector).tolist()), depth=self.depth+1, max_depth=self.max_depth))
            self.rects = rects

    def __str__(self):
        return "Hyperrectangular tree: dims {}, coords {}".format(self.n, self.coords)

class RectTree(object):
    def __init__(self, coords, depth=0, max_depth = 4):
        self.rects = None
        self.coords = coords
        self.depth = depth
        self.max_depth = max_depth

    def is_in_rectangle(self, coord):
        bound_x = (coord[0] <= self.coords[2]) and (coord[0] >= self.coords[0])
        bound_y = (coord[1] <= self.coords[3]) and (coord[1] >= self.coords[1])
        return (bound_x and bound_y)

    def get_cell(self, coord):
        if not self.is_in_rectangle(coord):
            return None
        elif self.rects == None:
            return self
        else:
            for r in self.rects:
                if r.is_in_rectangle(coord):
                    return r.get_cell(coord)

    def get_coords(self):
        return self.coords

    def subdivide(self):
        x0 = self.coords[0]
        y0 = self.coords[1]
        x1 = self.coords[2]
        y1 = self.coords[3]

        xm = (x0+x1)/2
        ym = (y0+y1)/2

        if self.rects == None and self.depth <= self.max_depth:
            self.rects = [RectTree((x0,y0, xm, ym), depth=self.depth+1), 
                            RectTree((xm, y0, x1, ym), depth=self.depth+1), 
                            RectTree((x0, ym, xm, y1), depth=self.depth+1), 
                            RectTree((xm, ym, x1, y1), depth=self.depth+1)]
    
    def __str__(self):
        return "Rectangle Tree: %d depth, coords (%d %d %d %d)" % (self.depth, self.coords[0], self.coords[1], self.coords[2], self.coords[3])

def draw_rect_tree(tree):
    '''draw_rect_tree

    tree -- either a RectTree or HyperRectTree where n = 2
    
    returns a pyplot figure of the cells 
    '''
    def draw_rect(rect, current_axis):
        if rect.rects == None:
            coords = rect.get_coords()
            x0 = coords[0]
            y0 = coords[1]
            width = (coords[2] - coords[0])
            height = (coords[3] - coords[1])
            current_axis.add_patch(Rectangle((x0, y0), width, height, fill=None, alpha=1))
        else:
            for r in rect.rects:
                draw_rect(r, current_axis)

    someX, someY = 0.5, 0.5
    fig = plt.figure()
    currentAxis = plt.gca()
    coords = tree.get_coords()
    plt.xlim(coords[0], coords[2])
    plt.ylim(coords[1], coords[3])
    draw_rect(tree, currentAxis)
    
    return fig


if __name__ == "__main__":
    #tree = HyperRectTree([.2, .4, 1, 1])
    #htree.subdivide()
    #htree.get_cell([0.6, 0.6]).subdivide()
    #htree.get_cell([0.6, 0.6]).subdivide()
    #print(htree.is_in_cell((1.75, 0.25)))
    #draw_rect_tree(htree)

    
    rtree = HyperRectTree([0, 0, 2, 1])

    rtree.get_even_per_cell(3)


    print(rtree.get_cell((1.75, 0.25)))

    pts = [(0.1, 0.1), (0.15, 0.15), (0.2, 0.2)]
    for pt in pts:
        print(pt)
        for i in range(0, 2):
            cell = rtree.get_cell(pt)
            cell.subdivide()
    print(rtree.get_cell((0.75, 0.25)))
    f = draw_rect_tree(rtree)
    plt.show()
    
    