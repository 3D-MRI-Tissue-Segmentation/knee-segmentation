# Renders generate toy data output in Panda3d from random voxel generated data,
# which is a 4D numpy array of 3 spatial dimensions and 1 color

# Rought outline
# - constrain viewing region to volume_array size + some margin
# - add a node at each voxel location and store the color
# - Turn every node into a voxel with that color on each face 


from direct.showbase.ShowBase import ShowBase
import numpy as np

base = ShowBase()
base.disableMouse()
base.camera.setPos(0, -10, 0)

# def initializeSim()   
#     """ Sets world settings like text, cam position etc. """
#     base.disableMouse()
#     base.camera.setPos(0, -10, 0)
# end initializeSim()


def makeSquare(x1, y1, z1, x2, y2, z2):       # This is straight copied off the Panda3d 'procedural cube' example https://github.com/panda3d/panda3d/blob/master/samples/procedural-cube/main.py#L11
    format = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData('square', format, Geom.UHDynamic)

    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    color = GeomVertexWriter(vdata, 'color')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    # make sure we draw the sqaure in the right plane
    if x1 != x2:
        vertex.addData3(x1, y1, z1)
        vertex.addData3(x2, y1, z1)
        vertex.addData3(x2, y2, z2)
        vertex.addData3(x1, y2, z2)

        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z2 - 1))
        normal.addData3(normalized(2 * x1 - 1, 2 * y2 - 1, 2 * z2 - 1))

    else:
        vertex.addData3(x1, y1, z1)
        vertex.addData3(x2, y2, z1)
        vertex.addData3(x2, y2, z2)
        vertex.addData3(x1, y1, z2)

        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z2 - 1))
        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z2 - 1))

    # adding different colors to the vertex for visibility
    color.addData4f(1.0, 0.0, 0.0, 1.0)
    color.addData4f(0.0, 1.0, 0.0, 1.0)
    color.addData4f(0.0, 0.0, 1.0, 1.0)
    color.addData4f(1.0, 0.0, 1.0, 1.0)

    texcoord.addData2f(0.0, 1.0)
    texcoord.addData2f(0.0, 0.0)
    texcoord.addData2f(1.0, 0.0)
    texcoord.addData2f(1.0, 1.0)

    # Quads aren't directly supported by the Geom interface
    # you might be interested in the CardMaker class if you are
    # interested in rectangle though
    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 3)
    tris.addVertices(1, 2, 3)

    square = Geom(vdata)
    square.addPrimitive(tris)
    return square


def getFacesNode(x,y,z): 
    """ Returns node with 6 cube face sides """
    node_name = 'faces' + str(x+y+z)
    faces_node = GeomNode(node_name)

    # Note: it isn't particularly efficient to make every face as a separate Geom.
    # instead, it would be better to create one Geom holding all of the faces.
    # Since xyz are centers of the voxel, getting the vertices means we treat the xyz's as the origin

    square0 = makeSquare(x-1, y-1, z-1, x+1, y-1, z+1)
    square1 = makeSquare(x-1, y+1, z-1, x+1, y+1, z+1)
    square2 = makeSquare(x-1, y+1, z+1, x+1, y-1, z+1)
    square3 = makeSquare(x-1, y+1, z-1, x+1, y-1, z-1)
    square4 = makeSquare(x-1, y-1, z-1, x-1, y+1, z+1)
    square5 = makeSquare(x+1, y-1, z-1, x+1, y+1, z+1)

    faces_node.addGeom(square0)
    faces_node.addGeom(square1)
    faces_node.addGeom(square2)
    faces_node.addGeom(square3)
    faces_node.addGeom(square4)
    faces_node.addGeom(square5)

    return faces_node


def create_voxel(x,y,z, color):    # Where xyz are the center of the voxel
    """ Creates and renders voxel """
    # All you have to do is get all 6 faces of the voxel and attach them to a node

    faces_node = getFacesNode(x,y,z)    # returns Geoms 6 squares
    cube = render.attachNewNode(faces_node)

    # OpenGl by default only draws "front faces" (polygons whose vertices are
    # specified CCW).
    cube.setTwoSided(True)



def get_random_xyz(self):
        x = randint(0, self.width-1)
        y = randint(0, self.height-1)
        z = randint(0, self.depth-1)
        return x, y, z

# volume = np.zeros([2, 2, 2, 3], dtype=float)
volume = numpy.random.rand(2, 2, 2, 3)
create_voxel(volume[:,:,:, 0], volume[0,0,0,:])


cube.hprInterval(6, (360, 360, 360)).loop()         # This rotates the cube lol




base.run()

