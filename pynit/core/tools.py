from handlers import sys, os
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

def itksnap(procobj, input_path, temp_path=None):

    def scan_update(*args):
        if procobj._sessions:
            scan_dropdown.options = list_of_items[ses_toggle.value][sub_toggle.value]
        else:
            scan_dropdown.options = list_of_items[sub_toggle.value]

    def sess_update(*args):
        ses_toggle.options = list_of_items[sub_toggle.value]

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
    sub_toggle = widgets.SelectionSlider(options=sorted(procobj._subjects), description='Subjects:',
                                         layout=widgets.Layout(width='600px', ))
    if procobj._sessions:
        ses_toggle = widgets.Dropdown(options=sorted(procobj._sessions), description='Sessions:',
                                      layout=widgets.Layout(width='600px'))
        scan_dropdown = widgets.RadioButtons(options=list_of_items[sub_toggle.value][ses_toggle.value],
                                             description='Scans:',
                                             layout=widgets.Layout(width='600px'))
        img_path = os.path.join(input_path, sub_toggle.value, ses_toggle.value, scan_dropdown.value)
        ses_toggle.observe(sess_update, 'value')
    else:
        scan_dropdown = widgets.RadioButtons(options=list_of_items[sub_toggle.value],
                                             description='Scans:',
                                             layout=widgets.Layout(width='600px'))
        img_path = os.path.join(input_path, sub_toggle.value, scan_dropdown.value)
    sub_toggle.observe(scan_update, 'value')
    launcher = widgets.Button(description='Launch ITK-snap', layout=widgets.Layout(width='600px'))

    def run_itksnap(sender):
        if temp_path:
            cmd = 'itksnap -g {} -s {}'.format(temp_path, img_path)
        else:
            cmd = 'itksnap -g {}'.format(img_path)
        methods.shell(cmd)

    launcher.on_click(run_itksnap)
    return widgets.VBox([sub_toggle, scan_dropdown, launcher])