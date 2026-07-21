import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cs
import time
import h5py
import re

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from PIL import ImageTk
import numpy as np
from pathlib import Path

import rydcalc

# fix deprecated function calls for newer versions of numpy and scipy
rydcalc.large_search.fix_libs()

h5_folder = r'C:\Users\SemeghiniLab\Harvard University Dropbox\Haley Nguyen\RbYb lab\07 Code\rydcalc\results\7_16_2026'



class ResultBrowser:

    def __init__(self, folder, Bz):

        self.h5_files = sorted(Path(folder).glob('*.h5'))

        # get all configurations stored in the h5 file
        self.configs = {h5_filename: [] for h5_filename in self.h5_files}

        def find_configs(h5_filename, name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith(')'):
                self.configs[h5_filename].append(name)

        for h5_filename in self.h5_files:
            with h5py.File(h5_filename, 'r') as h5_file:
                h5_file.visititems(lambda name, obj: find_configs(h5_filename, name, obj))

        self.Bz = Bz # [Gauss]
        # use these later
        self.Ez = 0  # [V/cm] 
        self.theta = 0

        self.gather_results()
        self.order = list(range(len(self.results)))
        self.index = 0

        self.root = tk.Tk()
        self.root.title('Rydberg Interaction Browser')
        self.root.geometry('1500x900')

        #---- Main layout ----#

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True)

        self.left = ttk.Frame(paned)
        self.right = ttk.Frame(paned, width=350)

        paned.add(self.left, weight=4)
        paned.add(self.right, weight=1)

        #---- Image display ----#

        self.image_label = ttk.Label(self.left)
        self.image_label.pack(fill='both', expand=True)

        self.photo = None
        self.left.bind('<Configure>', self._resize_image)

        #---- Metadata ----#

        ttk.Label(self.right, text='Configuration', font=('Arial', 11, 'bold')).pack(anchor='w', padx=5, pady=(5, 0))

        self.config_label = ttk.Label(self.right, text='', justify='left', wraplength=330)
        self.config_label.pack(fill='x', padx=5)

        ttk.Separator(self.right).pack(fill='x', pady=5)

        ttk.Label(self.right, text='Options', font=('Arial', 11, 'bold')).pack(anchor='w', padx=5)

        self.opts_box = ScrolledText(self.right, height=15, width=40)
        self.opts_box.pack(fill='both', padx=5)

        ttk.Separator(self.right).pack(fill='x', pady=5)

        ttk.Label(self.right, text='Coefficients', font=('Arial', 11, 'bold')).pack(anchor='w', padx=5)

        self.coef_box = ScrolledText(self.right, height=8, width=40)
        self.coef_box.pack(fill='both', expand=True, padx=5)

         #---- Comments ----#

        ttk.Separator(self.right).pack(fill='x', pady=5)

        ttk.Label(self.right, text='Comments', font=('Arial', 11, 'bold')).pack(anchor='w', padx=5)

        self.comment_box = ScrolledText(self.right, height=6, width=40)
        self.comment_box.pack(fill='x', padx=5)

        ttk.Button(self.right, text='Save Comment', command=self.save_comment).pack(pady=5)

        self.comment_box.delete('1.0', tk.END)

        self.flag_var = tk.BooleanVar()

        ttk.Checkbutton(
            self.right,
            text='Flag result',
            variable=self.flag_var,
            command=self.save_flag
        ).pack(anchor='w', padx=5)

        def toggle_flag(event):
            # toggle between 1 (checked) and 0 (unchecked)
            current_val = self.flag_var.get()
            new_val = 0 if current_val == 1 else 1
            self.flag_var.set(new_val)
            self.save_flag()

        self.root.bind("f", toggle_flag)


        #---- Sort controls ----#

        sort_frame = ttk.Frame(self.root)
        sort_frame.pack(fill='x', padx=5)

        ttk.Label(sort_frame, text='Sort:').pack(side='left')
        ttk.Button(sort_frame, text='C6d ↓', command=self.sort_c6d_high).pack(side='left', padx=2)
        ttk.Button(sort_frame, text='C6d ↑', command=self.sort_c6d_low).pack(side='left', padx=2)
        ttk.Button(sort_frame, text='C3d ↓', command=self.sort_c3d_high).pack(side='left', padx=(12,2))
        ttk.Button(sort_frame, text='C3d ↑', command=self.sort_c3d_low).pack(side='left', padx=2)

        self.show_flagged_only = tk.BooleanVar()
        self.hide_seen = tk.BooleanVar()

        ttk.Checkbutton(
            sort_frame,
            text='Show flagged only',
            variable=self.show_flagged_only,
            command=self.update_filter
        ).pack(side='left', padx=20)

        ttk.Checkbutton(
            sort_frame,
            text='Hide seen',
            variable=self.hide_seen,
            command=self.update_filter
        ).pack(side='left', padx=20)

        # allow user to sort by specific interaction
        options = ['All Interactions', 'Yb', 'Rb', 'Yb+Rb']
        self.combo = ttk.Combobox(sort_frame, values=options, state='readonly')
        self.combo.set('All Interactions')  # default placeholder text
        self.combo.pack(side='left', padx=20)
        self.combo.bind("<<ComboboxSelected>>", self.update_filter)

        # allow user to clear list of interactions marked as seen
        ttk.Button(sort_frame, text='Clear Seen', command=self.reset_seen).pack(side='left', padx=2)

        #---- Navigation ----#

        bottom = ttk.Frame(self.root)
        bottom.pack(fill='x')

        ttk.Button(bottom, text='<< Previous', command=self.prev_result).pack(side='left', padx=5, pady=5)
        self.root.bind("<Left>", lambda event: self.prev_result())
        ttk.Button(bottom, text='Next >>', command=self.next_result).pack(side='left')
        self.root.bind("<Right>", lambda event: self.next_result())

        self.slider = tk.Scale(bottom, from_=0, to=max(0, len(self.results)-1),
                               orient='horizontal', command=self.slider_changed, length=500)
        self.slider.pack(side='left', fill='x', expand=True, padx=20)

        self.status = ttk.Label(bottom)
        self.status.pack(side='right', padx=10)

        if self.results:
            self.show_result(0)

    def gather_results(self):
        # filter the interactions by field strength and angle

        self.results = []

        for h5_filename in self.h5_files:
            with h5py.File(h5_filename, 'r') as h5_file:
                for config in self.configs[h5_filename]:
                    _, current_Bz_str, current_Ez_str, current_theta_str = config.split('>')[-1].split(',')
                    current_Bz = float(re.sub(r'[^0-9.]', '', current_Bz_str.replace('np.float64', '')))

                    if current_Bz == self.Bz:
                        fig, coef_array, opts_dict = rydcalc.load_fig_from_h5(h5_file, config)

                        self.results.append({
                            'file': h5_filename,
                            'config': config,
                            'fig': fig,
                            'coef': coef_array,
                            'opts': opts_dict,
                        })

    def reset_seen(self):
        # marks all results as not seen
        for h5_filename in self.h5_files:
            with h5py.File(h5_filename, 'r+') as h5_file:
                if 'seen' in h5_file:
                    del h5_file['seen']

    def mark_seen(self, result):
        dt = h5py.string_dtype(encoding='utf-8')
        with h5py.File(result['file'], 'r+') as h5_file:
            if not 'seen' in h5_file:
                h5_file.create_dataset('seen', data=[result['config']], dtype=dt)

            else:
                seen = h5_file['seen'].asstr()[:]
                if not result['config'] in seen:
                    seen = np.append(seen, result['config'])
                    del h5_file['seen']
                    h5_file.create_dataset('seen', data=seen, dtype=dt)



    #---- Sorting ----#

    def sort_by(self, coef_index, reverse=False):
        self.order.sort(key=lambda i: self.results[i]['coef'][coef_index], reverse=reverse)
        self.show_result(0)

    def sort_c6d_high(self):
        self.sort_by(0, True)

    def sort_c6d_low(self):
        self.sort_by(0, False)

    def sort_c3d_high(self):
        self.sort_by(2, True)

    def sort_c3d_low(self):
        self.sort_by(2, False)

    def update_filter(self, event=None):
        self.order.clear()

        interaction_selection = self.combo.get()

        for h5_filename in self.h5_files:
            with h5py.File(h5_filename, 'r') as h5:
                for i, result in enumerate(self.results):
                    if h5_filename == result['file']:

                        interaction_num_yb = result['config'].count('Yb')
                        correct_interaction = (interaction_selection == 'Yb' and interaction_num_yb == 2) \
                                        or (interaction_selection == 'Rb' and interaction_num_yb == 0) \
                                        or (interaction_selection == 'Yb+Rb' and interaction_num_yb == 1) \
                                        or (interaction_selection == 'All Interactions')

                        flagged = 'flags' in h5 and result['config'] in h5['flags'].asstr()[:]
                        seen = 'seen' in h5 and result['config'] in h5['seen'].asstr()[:]

                        if (flagged or not self.show_flagged_only.get())  and (not seen or not self.hide_seen.get()) and correct_interaction:
                            self.order.append(i)

        self.slider.configure(to=max(len(self.order)-1,0))
        if len(self.order) > 0:
            self.show_result(0)
        else:
            self.image_label.configure(image='')
            self.config_label.configure(text='No results')

    #---- Navigation ----#

    def slider_changed(self, value):
        self.show_result(int(value))

    def next_result(self):
        if self.index < len(self.order)-1:
            self.show_result(self.index+1)

    def prev_result(self):
        if self.index > 0:
            self.show_result(self.index-1)

    #---- Image ----#

    def _resize_image(self, event=None):
        if not self.results:
            return

        img = self.results[self.order[self.index]]['fig']

        display = img.copy()

        if event is None:
            w, h = 900, 900
        else:
            w = max(event.width, 10)
            h = max(event.height, 10)

        display.thumbnail((w, h))

        self.photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.photo)

    #---- Flagging ----#

    def save_flag(self):
        print('updating flag...')
        result = self.results[self.order[self.index]]

        dt = h5py.string_dtype(encoding='utf-8')
        with h5py.File(result['file'], 'r+') as h5_file:
            if not 'flags' in h5_file and self.flag_var.get():
                h5_file.create_dataset('flags', data=[result['config']], dtype=dt)
                print('adding new true flag')

            elif self.flag_var.get():
                print('adding flag to existing dataset')
                flags = h5_file['flags'].asstr()[:]
                if not result['config'] in flags:
                    flags = np.append(flags, result['config'])
                    del h5_file['flags']
                    h5_file.create_dataset('flags', data=flags, dtype=dt)

            else:
                print('removing flag')
                flags = h5_file['flags'].asstr()[:]
                if result['config'] in flags:
                    flags = flags[flags != result['config']]
                    del h5_file['flags']
                    h5_file.create_dataset('flags', data=flags, dtype=dt)


    def load_flag(self):
        result = self.results[self.order[self.index]]
        self.flag_var.set(False)

        with h5py.File(result['file'], 'r') as h5:
            if 'flags' in h5 and result['config'] in h5['flags'].asstr()[:]:
                self.flag_var.set(True)

    #--- Comments ---#

    def save_comment(self):
        import h5py
        import numpy as np

        result = self.results[self.order[self.index]]
        config = result['config']

        comment = self.comment_box.get('1.0', tk.END).rstrip()

        with h5py.File(result['file'], 'a') as h5:

            name = f'{config}_comments'

            if name in h5:
                del h5[name]

            h5.create_dataset(
                name,
                data=np.bytes_(comment)
            )

        self.status.configure(text=f'Saved comment ({self.index+1}/{len(self.order)})')

    #---- Display ----#

    def show_result(self, idx):
        self.index = idx

        if idx < 1: return

        result = self.results[self.order[idx]]
        self.mark_seen(result)

        self._resize_image()

        self.config_label.configure(text=result['config'])

        self.opts_box.delete('1.0', tk.END)
        for k, v in sorted(result['opts'].items()):
            self.opts_box.insert(tk.END, f'{k:20s} : {v}\n')

        self.coef_box.delete('1.0', tk.END)

        labels = ['c6d', 'c6e', 'c3d', 'c3e']
        coef = np.asarray(result['coef'])

        if coef.ndim == 1:
            for label, value in zip(labels, coef):
                self.coef_box.insert(tk.END, f'{label:>4s} : {value:.6e}\n')
        else:
            self.coef_box.insert(tk.END, np.array2string(coef, precision=6))

        if self.slider.get() != idx:
            self.slider.set(idx)
        
        self.status.configure(text=f'{idx+1} / {len(self.order)}')

        self.load_flag()

        self.comment_box.delete('1.0', tk.END)

        result = self.results[self.order[self.index]]
        config = result['config']

        with h5py.File(result['file'], 'r') as h5:

            name = f'{config}_comments'

            if name not in h5:
                return

            comment = h5[name][()]

            if isinstance(comment, bytes):
                comment = comment.decode('utf-8')

            self.comment_box.insert('1.0', comment)

    #---- Main loop ----#

    def run(self):
        self.root.mainloop()


browser = ResultBrowser(h5_folder, Bz=10)
browser.run()