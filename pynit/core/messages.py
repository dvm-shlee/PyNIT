import warnings

class Warning(object):
    @staticmethod
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return '%s: %s\n' % (category.__name__, message)

    @staticmethod
    def deprecated():
        warnings.formatwarning = Warning.warning_on_one_line
        warnings.simplefilter("default")
        warnings.warn("pynit.Preprocess() will be deprecated soon. Please use pynit.Process() instead.", DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter("ignore")


class Notice(object):
    class MethodNotActivated(Exception):
        """MethodNotActivated"""
        pass

class Errors(object):
    """ Correction of CustomExceptions
    """
    class PackageUpdateFailure(EnvironmentError):
        """PackageUpdateFailure"""
        pass

    class InputTypeError(TypeError):
        """InputTypeError"""
        pass

    class InputValueError(ValueError):
        """InputValueError"""
        pass

    class UpdateAttributesFailed(TypeError):
        """UPdateAttributesFailed"""
        pass

    class InputDataclassError(IndexError):
        """InputDataclassError"""
        pass

    class KeywordError(KeyError):
        """KeywordError"""
        pass

    class ProjectScanFailure(LookupError):
        """ProjectScanFailure"""
        pass

    class DependenceError(NameError):
        """DependenceError"""
        pass

    class InitiationFailure(ReferenceError):
        """InitiationFailure"""
        pass

    class MissingPipeline(EnvironmentError):
        """MissingPipeline"""
        pass

    class NoFilteredOutput(EnvironmentError):
        """NoFilteredOutput"""
        pass

class InputFileError(Exception):
    pass


class InputPathError(Exception):
    pass


class InputObjectError(Exception):
    pass


class PipelineNotSet(Exception):
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


class ImportItkFailure(Exception):
    pass


class ArgumentsOverlapped(Exception):
    pass


class CommandExecutionFailure(Exception):
    pass


class NotExistingCommand(Exception):
    pass


class UnableInterfaceCommand(Exception):
    pass


class ObjectMismatch(Exception):
    pass
