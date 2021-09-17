class Image:
    def to_image(self, filename=None, *args, **kwargs):
        """Read the data and generate an image writing to the given filename.

        The filetype is automatically deduced from the filename; possible
        are common raster (png, jpg) and vector (svg, pdf) formats.

        If no filename is provided a default filename is deduced from the
        name of the class and the picture has png format.

        The other arguments are passed on to the plot function without changes,
        please check that documentation for the specifics."""
        fig = self.to_plotly(*args, **kwargs)
        default = f"{self.__class__.__name__.lower()}.png"
        filename = filename if filename is not None else default
        fig.write_image(self._path / filename)
