import sys, os
import messages
import methods
jupyter_env = False
try:
    if len([key for key in sys.modules.keys() if key == 'ipykernel']):
        from ipywidgets import widgets
        jupyter_env = True
except:
    pass


def afni(procobj, input_path):
    """Launch afni

    :param procobj:
    :param input_path:
    :return:
    """
    groups = procobj._subjects[:]
    groups_path = map(os.path.join, [input_path] * len(groups), groups)
    out, err = methods.shell('afni {}'.format(str(' '.join(groups_path))))
    return out, err


def itksnap(procobj, input_path, temp_path=None):
    """ run itksnap for given pass

    :param procobj:
    :param input_path:
    :param temp_path:
    :return:
    """


    def check_temppath(*args):
        if temp_path:
            if os.path.isfile(temp_path):
                output = temp_path
            else:
                if os.path.exists(temp_path):
                    output = os.path.join(temp_path, *args)
                else:
                    if len(args) == 3:
                        output =  os.path.join(procobj.prj.path, procobj.prj.ds_type[0],
                                               args[0], args[1], temp_path, args[2])
                    elif len(args) == 2:
                        output = os.path.join(procobj.prj.path, procobj.prj.ds_type[0],
                                              args[0], temp_path, args[1])
                    else:
                        output = None
                        methods.raiseerror(messages.Errors.InputValueError, 'Wrong arguments!')
        else:
            output = temp_path
        return output

    def run_itksnap(sender):
        # Run itksnap
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
        methods.shell(cmd)

    # Generate dictionary
    list_of_items = {}
    img_ext = procobj.prj.img_ext
    for subj in procobj._subjects:
        subj_path = os.path.join(input_path, subj)
        if procobj._sessions:
            list_of_items[subj] = {}
            try:
                for sess in [s for s in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, s))]:
                    sess_path = os.path.join(subj_path, sess)
                    scan_list = []
                    for scan in os.listdir(sess_path):
                        if all(ext in scan for ext in img_ext):
                            scan_list.append(scan)
                        list_of_items[subj][sess] = scan_list
            except:
                pass
        else:
            scan_list = []
            try:
                for scan in os.listdir(subj_path):
                    if all(ext in scan for ext in img_ext):
                        scan_list.append(scan)
                    list_of_items[subj] = scan_list
            except:
                pass

    # Subject dropdown widget prep
    sub_toggle = widgets.Dropdown(options=sorted(list_of_items.keys()), description='Subjects:',
                                  layout=widgets.Layout(width='600px', ))
    ses_toggle = None

    # If multisession, prep session dropdown widget

    def scan_update(*args):
        # Update list of files
        if procobj._sessions:
            scan_dropdown.options = sorted(list_of_items[sub_toggle.value][ses_toggle.value])
        else:
            scan_dropdown.options = sorted(list_of_items[sub_toggle.value])
        # Update widget values
        if scan_dropdown.value not in scan_dropdown.options:
            scan_dropdown.value = scan_dropdown.options[0]

    if procobj._sessions:
        def sess_update(*args):
            ses_toggle.options = sorted(list_of_items[sub_toggle.value].keys())
        ses_toggle = widgets.Dropdown(options=sorted(list_of_items[sub_toggle.value].keys()), description='Sessions:',
                                      layout=widgets.Layout(width='600px', ))
        scan_dropdown = widgets.Dropdown(options=sorted(list_of_items[sub_toggle.value][ses_toggle.value]),
                                             description='Scans:', layout=widgets.Layout(width='600px', ))
        sub_toggle.observe(sess_update, 'value') # if change value of sub_toggle, run ses_update
        sub_toggle.observe(scan_update, 'value') # ...
        ses_toggle.observe(scan_update, 'value')
    else:
        scan_dropdown = widgets.Dropdown(options=sorted(list_of_items[sub_toggle.value]),
                                             description='Scans:',
                                             layout=widgets.Layout(width='600px', ))
        sub_toggle.observe(scan_update, 'value')
        scan_dropdown.observe(scan_update, 'value')

    launcher = widgets.Button(description='Launch ITK-snap', layout=widgets.Layout(width='600px', ))
    launcher.on_click(run_itksnap)
    if procobj._sessions:
        return widgets.VBox([sub_toggle, ses_toggle, scan_dropdown, launcher])
    else:
        return widgets.VBox([sub_toggle, scan_dropdown, launcher])