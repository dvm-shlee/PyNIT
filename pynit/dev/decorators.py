from functools import wraps

def loop_over_subjects(prj):
    subjects = prj.subjects[:]
    def true_decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            for subj in subjects:
                files = prj(args[0], subj, *args[1:], **kwargs)
                for i, finfo in files:
                    print finfo.Filename
                r = f(*args, **kwargs)
            return r
        return wrapped
    return true_decorator