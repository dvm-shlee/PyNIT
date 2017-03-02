from handlers import sys, os, messages
import methods
try:
    if len([key for key in sys.modules.keys() if key == 'ipykernel']):
        # from tqdm import tqdm_notebook as progressbar
        from ipywidgets import widgets
    else:
        # from tqdm import tqdm as progressbar
        pass
except:
    pass

def afni(procobj, input_path, tmpobj=None):
    """Launch afni

    :param procobj:
    :param input_path:
    :return:
    """
    groups = procobj._subjects[:]
    groups_path = map(os.path.join, [input_path] * len(groups), groups)
    if tmpobj:
        groups_path += [os.path.dirname(tmpobj.image.get_filename())]
    out, err = methods.shell('afni {}'.format(str(' '.join(groups_path))))
    return out, err

def itksnap(procobj, input_path, temp_path=None):
    """ run itksnap for given pass

    :param procobj:
    :param input_path:
    :param temp_path:
    :return:
    """

    def scan_update(*args):
        if procobj._sessions:
            scan_dropdown.options = list_of_items[ses_toggle.value][sub_toggle.value]
        else:
            scan_dropdown.options = list_of_items[sub_toggle.value]

    def sess_update(*args):
        ses_toggle.options = list_of_items[sub_toggle.value]

    def check_temppath(*args):
        if temp_path:
            if os.path.isfile(temp_path):
                output = temp_path
            else:
                if os.path.exists(temp_path):
                    output = os.path.join(temp_path, *args)
                else:
                    if len(args) == 3:
                        output =  os.path.join(procobj._prjobj.path, procobj._prjobj.ds_type[0],
                                               args[0], args[1], temp_path, args[2])
                    elif len(args) == 2:
                        output = os.path.join(procobj._prjobj.path, procobj._prjobj.ds_type[0],
                                              args[0], temp_path, args[1])
                    else:
                        output = None
                        methods.raiseerror(messages.Errors.InputValueError, 'Wrong arguments!')
        else:
            output = temp_path
        return output

    list_of_items = {}
    img_ext = procobj._prjobj.img_ext
    for subj in procobj._subjects:
        subj_path = os.path.join(input_path, subj)
        if procobj._sessions:
            list_of_items[subj] = {}
            for sess in [s for s in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, s))]:
                sess_path = os.path.join(subj_path, sess)
                scan_list = []
                for scan in os.listdir(sess_path):
                    if any(ext in scan for ext in img_ext):
                        scan_list.append(scan)
                    list_of_items[subj][sess] = scan_list
        else:
            scan_list = []
            for scan in os.listdir(subj_path):
                if any(ext in scan for ext in img_ext):
                    scan_list.append(scan)
                list_of_items[subj] = scan_list
    sub_toggle = widgets.Dropdown(options=sorted(procobj._subjects), description='Subjects:',
                                      layout=widgets.Layout(width='600px', ))
    if procobj._sessions:
        ses_toggle = widgets.Dropdown(options=sorted(procobj._sessions), description='Sessions:',
                                      layout=widgets.Layout(width='600px'))
        scan_dropdown = widgets.RadioButtons(options=list_of_items[sub_toggle.value][ses_toggle.value],
                                             description='Scans:',
                                             layout=widgets.Layout(width='600px'))

        ses_toggle.observe(sess_update, 'value')
    else:
        scan_dropdown = widgets.RadioButtons(options=list_of_items[sub_toggle.value],
                                             description='Scans:',
                                             layout=widgets.Layout(width='600px'))

    sub_toggle.observe(scan_update, 'value')
    launcher = widgets.Button(description='Launch ITK-snap', layout=widgets.Layout(width='600px'))

    def run_itksnap(sender):
        if procobj._sessions:
            img_path = os.path.join(input_path, sub_toggle.value, ses_toggle.value, scan_dropdown.value)
            main_path = check_temppath(sub_toggle.value, ses_toggle.value, scan_dropdown.value)
        else:
            img_path = os.path.join(input_path, sub_toggle.value, scan_dropdown.value)
            main_path = check_temppath(sub_toggle.value, scan_dropdown.value)
        if temp_path:
            cmd = 'itksnap -g {} -s {}'.format(main_path, img_path)
        else:
            cmd = 'itksnap -g {}'.format(img_path)
        # print(cmd)
        methods.shell(cmd)

    launcher.on_click(run_itksnap)
    return widgets.VBox([sub_toggle, scan_dropdown, launcher])