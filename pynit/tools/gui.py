import os
import messages
import methods
from .__init__ import widgets


def afni(procobj, input_path, tmpobj=None):
    """Launch afni

    :param procobj:
    :param input_path:
    :param tmpobj:
    :return:
    """
    groups = procobj._subjects[:]
    if procobj.prj.single_session:
        groups_path = map(os.path.join, [input_path] * len(groups), groups)
    else:
        sessions = procobj._sessions[:]
        groups_path = []
        for sess in sessions:
            groups_path.extend(map(os.path.join, [input_path] * len(groups), groups, [sess] * len(groups)))
    if tmpobj:
        print(tmpobj._path)
        out, err = methods.shell('afni {} -dset {}'.format(str(' '.join(groups_path)), tmpobj._path))
    else:
        out, err = methods.shell('afni {}'.format(str(' '.join(groups_path))))
    return out, err


def image_viewer(procobj, input_path, temp_path=None, viewer=None):
    """ run fsleyes for given pass

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

    def run_fsleyes(sender):
        # Run itksnap
        if procobj._sessions:
            img_path = os.path.join(input_path, sub_toggle.value, ses_toggle.value, scan_dropdown.value)
            main_path = check_temppath(sub_toggle.value, ses_toggle.value, scan_dropdown.value)
        else:
            img_path = os.path.join(input_path, sub_toggle.value, scan_dropdown.value)
            main_path = check_temppath(sub_toggle.value, scan_dropdown.value)
        if temp_path:
            cmd = 'fsleyes -ad {} {}'.format(main_path, img_path)
        else:
            cmd = 'fsleyes {}'.format(img_path)
        methods.shell(cmd)

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
    if viewer:
        if viewer == 'fsleyes':
            launcher = widgets.Button(description='Launch FSLeyes', layout=widgets.Layout(width='600px', ))
            launcher.on_click(run_fsleyes)
        elif viewer == 'itksnap':
            launcher = widgets.Button(description='Launch ITK-snap', layout=widgets.Layout(width='600px', ))
            launcher.on_click(run_itksnap)
        else:
            launcher = widgets.Button(description='No viewer', layout=widgets.Layout(width='600px', ))
    else:
        launcher = widgets.Button(description='No viewer', layout=widgets.Layout(width='600px', ))
    if procobj._sessions:
        return widgets.VBox([sub_toggle, ses_toggle, scan_dropdown, launcher])
    else:
        return widgets.VBox([sub_toggle, scan_dropdown, launcher])