class Image:
    def to_image(self, *args, filename=None, **kwargs):
        """Read the data and generate an image writing to the given filename.

        The filetype is automatically deduced from the filename; possible
        are common raster (png, jpg) and vector (svg, pdf) formats.
        If no filename is provided a default filename is deduced from the
        name of the class and the picture has png format.

        Note that the filename must be a keyword argument, i.e., you explicitly
        need to write *filename="name_of_file"* because the arguments are passed
        on to the plot function. Please check the documentation of that function
        to learn which arguments are allowed."""
        fig = self.to_plotly(*args, **kwargs)
        classname = self.__class__.__name__.lower().strip("_")
        filename = filename if filename is not None else f"{classname}.png"
        fig.write_image(self._path / filename)
