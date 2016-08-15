class InputFileError(Exception):
    pass


class InputPathError(Exception):
    pass


class InputObjectError(Exception):
    pass


class NoPipeline(Exception):
    pass


class ImageDimentionMismatched(Exception):
    pass


class NotExistingDataclass(Exception):
    pass


class FilterInputTypeError(Exception):
    pass


class UpdateFailed(Exception):
    pass


class EmptyProject(Exception):
    pass


class NoLabelFile(Exception):
    pass


class ReloadFailure(Exception):
    pass


class ImportItkFailure(Exception):
    pass


class ArgumentsOverlapped(Exception):
    pass


class CommandExecutionFailure(Exception):
    pass
