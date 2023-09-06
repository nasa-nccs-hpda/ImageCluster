import logging
import tqdm


# -----------------------------------------------------------------------------
# Class TqdmLoggingHandler
# -----------------------------------------------------------------------------
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# -----------------------------------------------------------------------------
# getBaseCmd
# -----------------------------------------------------------------------------
def getBaseCmd(dataset) -> str:

    return 'gdalwarp ' + \
        ' -multi' + \
        ' -s_srs "' + \
        dataset.GetSpatialRef().ExportToProj4() + \
        '"'
