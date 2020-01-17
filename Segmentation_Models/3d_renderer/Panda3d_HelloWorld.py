# https://docs.panda3d.org/1.10/python/introduction/tutorial/using-intervals-to-move-the-panda

# Mouse Actions: 
# Left Button: Pan left and right.

# Right Button: Move forwards and backwards.

# Middle Button: Rotate around the origin of the application.

# Right and Middle Buttons: Roll the point of view around the view axis.





from direct.showbase.ShowBase import ShowBase


class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)


app = MyApp()
app.run()
