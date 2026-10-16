"""Tk/ttk user interface. Works on Windows without Colab or internet."""
from __future__ import annotations

import copy
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from app_config import BC_LOOKUP, BC_OPTIONS, initial_config, load_json, save_json, validate


class LayerDialog(tk.Toplevel):
    """Edit a full OrthotropicLayer record without losing existing properties."""
    VALUES = (
        ('name', 'Qatlam nomi'), ('h', 'Qalinlik [m]'), ('theta', 'Burchak [gradus]'),
        ('E1', 'E1 [Pa]'), ('E2', 'E2 [Pa]'), ('G12', 'G12 [Pa]'),
        ('G13', 'G13 [Pa]'), ('G23', 'G23 [Pa]'), ('nu12', 'nu12'),
        ('rho', 'Zichlik [kg/m³]'), ('A', 'RK A [s^(-alpha)]'),
        ('alpha', 'RK alpha'), ('beta', 'RK beta [1/s]'))

    def __init__(self, parent, layer, title):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.result = None
        self.data = copy.deepcopy(layer)
        self.vars = {}
        form = ttk.Frame(self, padding=16)
        form.pack(fill='both', expand=True)
        for i, (key, label) in enumerate(self.VALUES):
            col = 0 if i < 7 else 2
            row = i if i < 7 else i-7
            ttk.Label(form, text=label).grid(row=row, column=col, sticky='w', padx=(0,8), pady=5)
            var = tk.StringVar(value=str(layer.get(key, '')))
            ttk.Entry(form, textvariable=var, width=21).grid(row=row, column=col+1, sticky='ew', padx=(0,22))
            self.vars[key] = var
        self.visco = tk.BooleanVar(value=layer.get('is_visco', False))
        ttk.Checkbutton(form, text='Rjanitsin–Koltunov: viskoelastik qatlam',
                        variable=self.visco).grid(row=7, column=0, columnspan=4, sticky='w', pady=(12,6))
        ttk.Label(form, text='Mustahkamlik mezoni').grid(row=8, column=0, sticky='w', pady=5)
        self.criterion = tk.StringVar(value=layer.get('criterion', 'none'))
        ttk.Combobox(form, width=21, textvariable=self.criterion, state='readonly',
                     values=('none', 'von_mises', 'max_stress_plane', 'max_stress')).grid(
                         row=8,column=1, sticky='ew')
        ttk.Label(form, text='Mustahkamliklar [Pa]: JSON').grid(
            row=9,column=0,columnspan=4,sticky='w',pady=(8,2))
        self.strengths = scrolledtext.ScrolledText(form, width=66,height=4,wrap='word')
        self.strengths.grid(row=10,column=0,columnspan=4,sticky='ew')
        self.strengths.insert('1.0', json.dumps(layer.get('strengths', {}),
                                               ensure_ascii=False, indent=2))
        ttk.Label(form, text='von_mises: yield; max_stress_plane: Xt,Xc,Yt,Yc,S12; '
                              'max_stress: shular + S13,S23', foreground='#596574',
                  wraplength=610).grid(row=11,column=0,columnspan=4,sticky='w',pady=(2,12))
        btns = ttk.Frame(form)
        btns.grid(row=12,column=0,columnspan=4,sticky='e')
        ttk.Button(btns,text='Bekor qilish', command=self.destroy).pack(side='right',padx=4)
        ttk.Button(btns,text='Saqlash',command=self.accept).pack(side='right',padx=4)
        self.bind('<Escape>', lambda _e:self.destroy())
        self.grab_set()
        self.wait_window(self)

    def accept(self):
        try:
            data = copy.deepcopy(self.data)
            for key, _ in self.VALUES:
                data[key] = self.vars[key].get().strip() if key == 'name' else float(self.vars[key].get())
            data['is_visco'] = bool(self.visco.get())
            data['criterion'] = self.criterion.get()
            strengths = json.loads(self.strengths.get('1.0','end').strip() or '{}')
            if not isinstance(strengths, dict):
                raise ValueError('Mustahkamliklar JSON obyekti bo\'lsin.')
            data['strengths'] = strengths
            if not data['is_visco']:
                data['A'] = 0.0
            import engine
            engine.OrthotropicLayer(**data)
            self.result = data
            self.destroy()
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            messagebox.showerror('Qatlam parametrlari', str(exc), parent=self)


class Application:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('CLPT + FSDT + RK | Kompozit plastinka tahlili')
        self.root.geometry('1260x800')
        self.root.minsize(980, 670)
        self.config_path = None
        self.current_output = None
        self.layers = []
        self.field_vars = {}
        self.bc_vars = {}
        self.proc = None
        self.log_offset = 0
        self.msgs = queue.Queue()
        self._picture = None
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
        self.style.configure('TButton', padding=(10, 6))
        self.style.configure('TNotebook.Tab', padding=(15,8))
        self.style.configure('Header.TLabel',font=('Segoe UI',16,'bold'))
        self.style.configure('Muted.TLabel', foreground='#53637a')
        self._make_menu()
        self._make_body()
        self.set_config(initial_config())
        self._message('Namunaviy qiymatlar tahrirlash uchun yuklandi. Hisoblash faqat '
                      '"Hisoblash" tugmasini bosganingizda boshlanadi.\n')
        self.root.after(100, self._consume_queue)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def _make_menu(self):
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu,tearoff=0)
        file_menu.add_command(label='Yangi / namunaviy parametrlar',command=self.load_example)
        file_menu.add_command(label='JSON konfiguratsiyani ochish... ',command=self.open_config)
        file_menu.add_command(label='JSON konfiguratsiyani saqlash... ',command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label='Oxirgi natijalar papkasi',command=self.open_output)
        file_menu.add_separator()
        file_menu.add_command(label='Chiqish',command=self.close)
        menu.add_cascade(label='Fayl',menu=file_menu)
        tasks = tk.Menu(menu,tearoff=0)
        tasks.add_command(label='Hisoblash',command=self.start)
        tasks.add_command(label='Hisoblashni to\'xtatish',command=self.stop)
        menu.add_cascade(label='Hisoblash', menu=tasks)
        menu.add_command(label='Dastur haqida',command=self.about)
        self.root.config(menu=menu)

    def _make_body(self):
        header = ttk.Frame(self.root, padding=(16,13,16,8))
        header.pack(fill='x')
        ttk.Label(header,text='CLPT + FSDT + Rjanitsin–Koltunov',style='Header.TLabel').pack(anchor='w')
        ttk.Label(header,text='Kompozit va viskoelastik plastinkalar • SI birliklar • Offline hisoblash',
                  style='Muted.TLabel').pack(anchor='w',pady=(4,0))
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both',expand=True,padx=13,pady=(0,8))
        self.tab_geom, self.tab_layers, self.tab_load, self.tab_time, self.tab_results = (
            ttk.Frame(self.tabs,padding=14) for _ in range(5))
        for name,tab in [('Geometriya va tayanchlar',self.tab_geom),
                         ('Qatlamlar',self.tab_layers),('Yuklanish',self.tab_load),
                         ('Hisob parametrlari',self.tab_time),('Natijalar',self.tab_results)]:
            self.tabs.add(tab,text=name)
        self._tab_geometry()
        self._tab_layers()
        self._tab_loading()
        self._tab_simulation()
        self._tab_report()
        controls = ttk.Frame(self.root,padding=(13,6,13,12))
        controls.pack(fill='x')
        self.status = tk.StringVar(value='Tayyor. Parametrlarni tekshiring va Hisoblash tugmasini bosing.')
        ttk.Label(controls,textvariable=self.status).pack(side='left',fill='x',expand=True)
        self.spinner = ttk.Progressbar(controls,mode='indeterminate',length=140)
        self.spinner.pack(side='right',padx=6)
        self.stop_button = ttk.Button(controls,text='To\'xtatish', command=self.stop,state='disabled')
        self.stop_button.pack(side='right',padx=4)
        self.run_button = ttk.Button(controls,text='▶ Hisoblash',command=self.start)
        self.run_button.pack(side='right',padx=4)
        ttk.Button(controls,text='JSON saqlash',command=self.save_config).pack(side='right',padx=4)

    @staticmethod
    def _section(parent, text):
        box = ttk.LabelFrame(parent,text=text,padding=14)
        return box

    def _entry(self,parent,key,label,row,column=0,width=17):
        ttk.Label(parent,text=label).grid(row=row,column=column,sticky='w',padx=(0,10),pady=7)
        var = tk.StringVar()
        self.field_vars[key] = var
        widget = ttk.Entry(parent,textvariable=var,width=width)
        widget.grid(row=row,column=column+1,sticky='ew',padx=(0,18),pady=7)
        return widget

    def _choice(self,parent,key,label,values,row,column=0,width=18):
        ttk.Label(parent,text=label).grid(row=row,column=column,sticky='w',padx=(0,10),pady=7)
        var=tk.StringVar()
        self.field_vars[key]=var
        combo=ttk.Combobox(parent,textvariable=var,values=values,state='readonly',width=width)
        combo.grid(row=row,column=column+1,sticky='ew',padx=(0,18),pady=7)
        return combo

    def _tab_geometry(self):
        container = ttk.Frame(self.tab_geom)
        container.pack(fill='both',expand=True)
        left = ttk.Frame(container)
        right = ttk.Frame(container)
        left.pack(side='left',fill='both',expand=True,padx=(0,10))
        right.pack(side='left',fill='both',expand=True)
        box = self._section(left,'Plastinka va diskretizatsiya')
        box.pack(fill='x',pady=(0,14))
        for idx, (key,label) in enumerate((('a','Uzunlik a [m]'),('b','Eni b [m]'),
                  ('nx','FSDT elementlar x'),('ny','FSDT elementlar y'),
                  ('clpt_nx','CLPT Ritz modalari x'),('clpt_ny','CLPT Ritz modalari y'))):
            self._entry(box,key,label,idx)
        ttk.Label(box,text='FSDT: elementlar soni; CLPT: Ritz modalari soni. '
                  'Ularni bir xil deb hisoblamang.',wraplength=490,
                  style='Muted.TLabel').grid(row=6,column=0,columnspan=2,sticky='w',pady=(10,0))
        found = self._section(left,'Elastik asos va tekislikdagi cheklov')
        found.pack(fill='x')
        self._entry(found,'kw','Winkler kw [N/m³]',0)
        self._entry(found,'kp','Pasternak kp [N/m]',1)
        self._choice(found,'in_plane','Sharnir uchun u/v',('fixed','normal','free'),2)
        ttk.Label(found,text='fixed = u,v cheklangan; normal = faqat normal komponent; '
                  'free = tekislikda erkin.',wraplength=490,
                  style='Muted.TLabel').grid(row=3,column=0,columnspan=2,sticky='w',pady=(9,0))
        bcs = self._section(right,'Chegaraviy shartlar')
        bcs.pack(fill='x',pady=(0,14))
        for row,(key,label) in enumerate((('left','Chap (x=0)'),('right','O\'ng (x=a)'),
                                          ('bottom','Pastki (y=0)'),('top','Yuqori (y=b)'))):
            ttk.Label(bcs,text=label).grid(row=row,column=0,sticky='w',pady=12,padx=(0,10))
            var=tk.StringVar()
            self.bc_vars[key]=var
            ttk.Combobox(bcs,textvariable=var,values=tuple(BC_OPTIONS),
                         state='readonly',width=21).grid(row=row,column=1,sticky='w')
        ttk.Label(bcs,text='S — sharnirli, C — qistirilgan, F — erkin',
                  style='Muted.TLabel').grid(row=4,column=0,columnspan=2,sticky='w',pady=(12,0))
        info=self._section(right,'Modelni tanlash')
        info.pack(fill='x')
        self._choice(info,'model','Hisoblash nazariyasi',('both','fsdt','clpt'),0)
        ttk.Label(info,text='both: ikkala usul mustaqil hisoblanadi va taqqoslanadi.\n'
                  'fsdt: MITC4 FEM • clpt: Rayleigh–Ritz',
                  style='Muted.TLabel',justify='left').grid(row=1,column=0,columnspan=2,
                                                              sticky='w',pady=(6,0))

    def _tab_layers(self):
        text=ttk.Frame(self.tab_layers)
        text.pack(fill='x',pady=(0,8))
        ttk.Label(text,text='Qatlamlarni PASTDAN YUQORIGA kiriting. Har bir qatlam '
                  'qalinligi, burchagi, ortotrop modullari va RK xotirasini alohida tahrirlash mumkin.',
                  wraplength=1020).pack(anchor='w')
        cols=('name','h','theta','E1','E2','rho','visco','criterion')
        frame=ttk.Frame(self.tab_layers)
        frame.pack(fill='both',expand=True)
        self.layer_tree=ttk.Treeview(frame,columns=cols,show='headings',height=14,selectmode='browse')
        widths=(180,96,95,145,145,95,110,130)
        labels=('Qatlam nomi','h [m]','θ [°]','E1 [Pa]','E2 [Pa]','ρ [kg/m³]','RK','Mezon')
        for col,width,label in zip(cols,widths,labels):
            self.layer_tree.heading(col,text=label)
            self.layer_tree.column(col,width=width,minwidth=80,anchor='center')
        sb=ttk.Scrollbar(frame,orient='vertical',command=self.layer_tree.yview)
        self.layer_tree.configure(yscrollcommand=sb.set)
        self.layer_tree.pack(side='left',fill='both',expand=True)
        sb.pack(side='right',fill='y')
        self.layer_tree.bind('<Double-1>',lambda _e:self.edit_layer())
        buttons=ttk.Frame(self.tab_layers)
        buttons.pack(fill='x',pady=(12,0))
        for text,fn in (('＋ Qatlam qo\'shish',self.add_layer),('✎ Tahrirlash',self.edit_layer),
                        ('− O\'chirish',self.remove_layer),('↑ Yuqoriga',lambda:self.move_layer(-1)),
                        ('↓ Pastga',lambda:self.move_layer(1))):
            ttk.Button(buttons,text=text,command=fn).pack(side='left',padx=(0,8))
        self.layer_count=tk.StringVar()
        ttk.Label(buttons,textvariable=self.layer_count).pack(side='right')

    def _tab_loading(self):
        main=ttk.Frame(self.tab_load)
        main.pack(fill='both',expand=True)
        left=self._section(main,'Yuk turi va intensivligi')
        right=self._section(main,'Yukning fazo/vaqt parametrlari')
        left.pack(side='left',fill='both',expand=True,padx=(0,12))
        right.pack(side='left',fill='both',expand=True)
        kind=self._choice(left,'kind','Yuk turi',('point','uniform','pulse','harmonic'),0)
        space=self._choice(left,'spatial','Yuk fazosi',('point','patch','uniform'),1)
        self._entry(left,'amplitude','Amplituda [N yoki Pa]',2)
        ttk.Label(left,text='Nuqtaviy/kontakt yuk: N (umumiy kuch).\n'
                  'Bir tekis taqsimlangan yuk: Pa.',style='Muted.TLabel',
                  justify='left').grid(row=3,column=0,columnspan=2,sticky='w',pady=(10,0))
        for idx,(key,label) in enumerate((('x','Yuk markazi x [m]'),('y','Yuk markazi y [m]'),
            ('patch_width','Kontakt eni [m]'),('patch_height','Kontakt bo\'yi [m]'),
            ('frequency_hz','Garmonik f [Hz]'),('phase_degrees','Faza [°]'),
            ('duration','Zarb davomiyligi [s]'),('start_time','Yuk boshlanishi [s]'),
            ('ramp_time','O\'sish vaqti [s]'),('rise_fraction','Triangle ulushi'),
            ('decay','Friedlander decay'))):
            self._entry(right,key,label,idx)
        self._choice(right,'profile','Zarb shakli',('half_sine','triangle','friedlander'),11)
        ttk.Label(right,text='Keraksiz parametrlar tanlangan yuk turida hisobga olinmaydi.',
                  style='Muted.TLabel',wraplength=450).grid(row=12,column=0,columnspan=2,
                                                              sticky='w',pady=(9,0))
        kind.bind('<<ComboboxSelected>>',self._kind_changed)
        space.bind('<<ComboboxSelected>>',self._kind_changed)

    def _tab_simulation(self):
        form=self._section(self.tab_time,'Vaqt va integratsiya sozlamalari')
        form.pack(anchor='nw',fill='x')
        for i,(key,label) in enumerate((('total_time','Hisoblash vaqti [s]'),
             ('dt','Vaqt qadami dt [s] (0 = avtomatik)'),('w0_mm','Boshlang\'ich w [mm]'),
             ('v0','Boshlang\'ich tezlik [m/s]'),
             ('kernel_tolerance','RK yadro aproksimatsiya xatosi'))):
            self._entry(form,key,label,i)
        self._choice(form,'memory_initial','Boshlang\'ich xotira',
                     ('instantaneous','relaxed'),5)
        self.field_vars['calculate_stress']=tk.BooleanVar()
        ttk.Checkbutton(form,text='FSDT qatlam kuchlanishlarini hisoblash',
                        variable=self.field_vars['calculate_stress']).grid(row=6,column=0,columnspan=2,
                                                                            sticky='w',pady=9)
        self.field_vars['record_energy']=tk.BooleanVar()
        ttk.Checkbutton(form,text='Energiya tarixini hisoblash (sekinlashtirishi mumkin)',
                        variable=self.field_vars['record_energy']).grid(row=7,column=0,columnspan=2,
                                                                          sticky='w',pady=9)
        self.field_vars['check_dt']=tk.BooleanVar(value=False)
        ttk.Checkbutton(form,text='dt/2 vaqt aniqligini qo\'shimcha tekshirish',
                        variable=self.field_vars['check_dt']).grid(row=8,column=0,columnspan=2,
                                                                     sticky='w',pady=9)
        ttk.Label(self.tab_time,text='Eslatma: bu chiziqli model. Mustahkamlik va '
                  'konvergentsiyani muhandislik etalonlari bilan alohida tekshiring.',
                  style='Muted.TLabel',wraplength=800).pack(anchor='w',pady=(20,0))
        dest=self._section(self.tab_time,'Natijalar uchun asosiy papka')
        dest.pack(fill='x',pady=(20,0))
        ttk.Label(dest,text='Manzil').grid(row=0,column=0,sticky='w',padx=(0,10))
        self.field_vars['result_root']=tk.StringVar(value=str(Path.home()/'Documents'/'CLPT_FSDT_RK'))
        ttk.Entry(dest,textvariable=self.field_vars['result_root'],width=65).grid(
            row=0,column=1,sticky='ew',padx=(0,9))
        ttk.Button(dest,text='Tanlash...',command=self.pick_folder).grid(row=0,column=2)
        dest.columnconfigure(1,weight=1)

    def _tab_report(self):
        head=ttk.Frame(self.tab_results)
        head.pack(fill='x',pady=(0,8))
        ttk.Button(head,text='Natijalar papkasini ochish',command=self.open_output).pack(side='left')
        ttk.Button(head,text='Hisobot grafikini ochish',command=self.open_picture).pack(side='left',padx=8)
        self.folder_lbl=tk.StringVar(value='Natijalar hali yo\'q')
        ttk.Label(head,textvariable=self.folder_lbl,style='Muted.TLabel').pack(side='left',padx=8)
        pane=ttk.Panedwindow(self.tab_results,orient='horizontal')
        pane.pack(fill='both',expand=True)
        left=ttk.Frame(pane)
        right=ttk.Frame(pane)
        pane.add(left,weight=2)
        pane.add(right,weight=3)
        ttk.Label(left,text='Hisoblash jurnali / xatoliklar').pack(anchor='w',pady=(0,6))
        self.log_box=scrolledtext.ScrolledText(left,wrap='word',width=48,height=25,
                                               font=('Consolas',10))
        self.log_box.pack(fill='both',expand=True)
        ttk.Label(right,text='Qisqa natijalar').pack(anchor='w',pady=(0,6))
        self.summary=tk.Text(right,height=9,wrap='word',font=('Consolas',10))
        self.summary.pack(fill='x')
        self.summary.config(state='disabled')
        ttk.Label(right,text='Grafik (saqlangan PNG)').pack(anchor='w',pady=(10,6))
        self.graph_label=ttk.Label(right,text='Hisoblashdan keyin grafik shu yerda chiqadi',
                                   anchor='center')
        self.graph_label.pack(fill='both',expand=True)
        self.graph_label.bind('<Configure>',lambda _e:self._draw_chart())

    def _kind_changed(self, _event=None):
        if self.field_vars['kind'].get() == 'uniform':
            self.field_vars['spatial'].set('uniform')

    def _message(self, message):
        self.log_box.insert('end',message)
        self.log_box.see('end')

    def _refresh_layer_table(self):
        self.layer_tree.delete(*self.layer_tree.get_children())
        for i, layer in enumerate(self.layers):
            self.layer_tree.insert('', 'end',iid=str(i),values=(
                layer.get('name',f'Qatlam {i+1}'),f"{layer['h']:.5g}",f"{layer['theta']:g}",
                f"{layer['E1']:.4g}",f"{layer['E2']:.4g}",f"{layer['rho']:g}",
                'Ha' if layer.get('is_visco',False) else 'Yo\'q',layer.get('criterion','none')))
        self.layer_count.set(f'Jami: {len(self.layers)} qatlam')

    def selected_index(self):
        selected=self.layer_tree.selection()
        return int(selected[0]) if selected else None

    def add_layer(self):
        base=copy.deepcopy(self.layers[-1]) if self.layers else initial_config()['layers'][0]
        base['name']=f'Qatlam {len(self.layers)+1}'
        dlg=LayerDialog(self.root,base,'Yangi qatlam')
        if dlg.result:
            self.layers.append(dlg.result)
            self._refresh_layer_table()
            self.layer_tree.selection_set(str(len(self.layers)-1))

    def edit_layer(self):
        idx=self.selected_index()
        if idx is None:
            messagebox.showinfo('Qatlam','Avval jadvaldan qatlamni tanlang.',parent=self.root)
            return
        dlg=LayerDialog(self.root,self.layers[idx],f'{idx+1}-qatlamni tahrirlash')
        if dlg.result:
            self.layers[idx]=dlg.result
            self._refresh_layer_table()
            self.layer_tree.selection_set(str(idx))

    def remove_layer(self):
        idx=self.selected_index()
        if idx is None:
            return
        if len(self.layers)==1:
            messagebox.showwarning('Qatlam','Kamida bitta qatlam bo\'lishi shart.',parent=self.root)
            return
        self.layers.pop(idx)
        self._refresh_layer_table()

    def move_layer(self,step):
        idx=self.selected_index()
        if idx is None or not 0 <= idx+step < len(self.layers):
            return
        self.layers[idx],self.layers[idx+step]=self.layers[idx+step],self.layers[idx]
        self._refresh_layer_table()
        self.layer_tree.selection_set(str(idx+step))

    def set_config(self,cfg):
        import engine
        cfg=engine.normalize_config(cfg)
        self.layers=copy.deepcopy(cfg['layers'])
        self._refresh_layer_table()
        for key in ('a','b','nx','ny','kw','kp'):
            self.field_vars[key].set(str(cfg.get(key, {'kw':0,'kp':0}.get(key,''))))
        self.field_vars['clpt_nx'].set(str(cfg.get('clpt',{}).get('nx',4)))
        self.field_vars['clpt_ny'].set(str(cfg.get('clpt',{}).get('ny',4)))
        self.field_vars['in_plane'].set(cfg.get('in_plane','fixed'))
        for key in ('left','right','bottom','top'):
            self.bc_vars[key].set(BC_LOOKUP[cfg['edges'][key]])
        self.field_vars['model'].set(cfg.get('_gui_model','both'))
        load=cfg.get('load',{})
        defaults=dict(kind='pulse', spatial='patch',amplitude=60,x=cfg['a']/2,y=cfg['b']/2,
            patch_width=cfg['a']/10,patch_height=cfg['b']/10,frequency_hz=100,
            phase_degrees=0,duration=.003,start_time=0,ramp_time=0,
            rise_fraction=.1,decay=2,profile='half_sine')
        defaults.update(load)
        for key in defaults:
            if key in self.field_vars:
                self.field_vars[key].set(str(defaults[key]))
        self.field_vars['spatial'].set('uniform' if defaults['kind']=='uniform' else
            'patch' if defaults['spatial']=='patch' else 'point')
        sim=cfg['simulation']
        for key,default in (('total_time',.04),('dt',0),('w0_mm',0),('v0',0),
                            ('kernel_tolerance',2e-5)):
            self.field_vars[key].set(str(sim.get(key,default) if sim.get(key,default) is not None else 0))
        self.field_vars['memory_initial'].set(sim.get('memory_initial','instantaneous'))
        self.field_vars['calculate_stress'].set(sim.get('calculate_stress',True))
        self.field_vars['record_energy'].set(sim.get('record_energy',False))
        self.field_vars['check_dt'].set(bool(cfg.get('_gui_check_dt',False)))

    def make_config(self):
        value=lambda key:float(self.field_vars[key].get().strip().replace(',','.'))
        integral=lambda key:int(self.field_vars[key].get().strip())
        cfg={'a':value('a'),'b':value('b'),'nx':integral('nx'),'ny':integral('ny'),
             'clpt':{'nx':integral('clpt_nx'),'ny':integral('clpt_ny')},
             'kw':value('kw'),'kp':value('kp'),
             'in_plane':self.field_vars['in_plane'].get(),
             'edges':{key:BC_OPTIONS[var.get()] for key,var in self.bc_vars.items()},
             'layers':copy.deepcopy(self.layers)}
        load={'kind':self.field_vars['kind'].get(),'amplitude':value('amplitude')}
        load['spatial'] = 'uniform' if load['kind']=='uniform' else self.field_vars['spatial'].get()
        if load['kind']=='point' and load['spatial']=='uniform':
            raise ValueError('Nuqtaviy yuk uchun uniform tanlanmaydi; "uniform" yuk turini tanlang.')
        if load['spatial'] != 'uniform':
            load.update(x=value('x'),y=value('y'))
            if load['spatial']=='patch':
                load.update(patch_width=value('patch_width'),patch_height=value('patch_height'))
        if load['kind']=='harmonic':
            load.update(frequency_hz=value('frequency_hz'),phase_degrees=value('phase_degrees'))
        if load['kind']=='pulse':
            load.update(duration=value('duration'),profile=self.field_vars['profile'].get(),
                        rise_fraction=value('rise_fraction'),decay=value('decay'))
        load.update(start_time=value('start_time'),ramp_time=value('ramp_time'))
        cfg['load']=load
        sim={'total_time':value('total_time'),'w0_mm':value('w0_mm'),
             'v0':value('v0'),'memory_initial':self.field_vars['memory_initial'].get(),
             'kernel_tolerance':value('kernel_tolerance'),
             'calculate_stress':bool(self.field_vars['calculate_stress'].get()),
             'record_energy':bool(self.field_vars['record_energy'].get())}
        if value('dt')>0: sim['dt']=value('dt')
        elif value('dt')<0: raise ValueError('dt manfiy bo\'lmasin.')
        cfg['simulation']=sim
        return cfg

    def load_example(self):
        if self.proc:
            return
        self.set_config(initial_config())
        self.status.set('Namunaviy sonlar qo\'yildi. Istagan qiymatni o\'zgartiring.')

    def open_config(self):
        if self.proc:
            messagebox.showwarning('Hisoblash','Avval hisobni to\'xtating.')
            return
        path=filedialog.askopenfilename(title='Parametrlarni ochish',
              filetypes=[('JSON konfiguratsiya','*.json'),('Barcha fayllar','*.*')])
        if not path: return
        try:
            cfg=load_json(path)
            validate(cfg, cfg.get('_gui_model','both'))
            self.set_config(cfg)
            self.config_path=Path(path)
            self.status.set(f'Yuklandi: {path}')
        except Exception as exc:
            messagebox.showerror('JSON xatosi',str(exc),parent=self.root)

    def save_config(self):
        try:
            cfg=self.make_config()
            validate(cfg,self.field_vars['model'].get())
        except Exception as exc:
            messagebox.showerror('Parametr xatosi',str(exc),parent=self.root)
            return
        path=filedialog.asksaveasfilename(title='Parametrlarni JSON sifatida saqlash',
              initialfile='mening_plastinkam.json',defaultextension='.json',
              filetypes=[('JSON','*.json')])
        if path:
            cfg['_gui_model']=self.field_vars['model'].get()
            cfg['_gui_check_dt']=bool(self.field_vars['check_dt'].get())
            save_json(cfg,path)
            self.config_path=Path(path)
            self.status.set(f'Saqlandi: {path}')

    def pick_folder(self):
        target=filedialog.askdirectory(title='Natijalar papkasini tanlash',
                                       initialdir=str(Path.home()))
        if target:self.field_vars['result_root'].set(target)

    def _launch_cmd(self, config_path, output, model):
        # Same executable can be a GUI parent or invisible worker.  For source
        # mode it executes the main.py file with current Python interpreter.
        if getattr(sys,'frozen',False):
            cmd=[sys.executable]
        else:
            cmd=[sys.executable,str(Path(__file__).resolve().with_name('main.py'))]
        return cmd+['--worker','--config',str(config_path),
                    '--output',str(output),'--model',model]

    def start(self):
        if self.proc is not None:
            return
        try:
            model=self.field_vars['model'].get()
            cfg=validate(self.make_config(),model)
            output_root=Path(self.field_vars['result_root'].get().strip()).expanduser()
            output_root.mkdir(parents=True,exist_ok=True)
            target=output_root/f"hisob_{datetime.now():%Y%m%d_%H%M%S_%f}"
            target.mkdir(parents=True,exist_ok=False)
            cfg['_gui_model']=model
            cfg['_gui_check_dt']=bool(self.field_vars['check_dt'].get())
            config_path=save_json(cfg,target/'kiritilgan_parametrlar.json')
            cmd=self._launch_cmd(config_path,target,model)
            if self.field_vars['check_dt'].get():cmd.append('--check-dt')
            # File-based logs survive PyInstaller --windowed: Windows GUI EXEs
            # are not guaranteed to have usable stdout/stderr handles.
            options={'stdout':subprocess.DEVNULL,'stderr':subprocess.DEVNULL,
                     'stdin':subprocess.DEVNULL}
            if sys.platform=='win32':
                options['creationflags']=subprocess.CREATE_NO_WINDOW
            self.proc=subprocess.Popen(cmd,**options)
            self.current_output=target
            self.log_offset=0
            self.folder_lbl.set(str(target))
            self.log_box.delete('1.0','end')
            self._message('Ishga tushdi: '+model.upper()+'\n')
            self._message('Natijalar: '+str(target)+'\n\n')
            self.status.set('Hisoblash davom etmoqda...')
            self.run_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.spinner.start(15)
            self.tabs.select(self.tab_results)
            p=self.proc
            threading.Thread(target=self._watch_worker,args=(p,),daemon=True).start()
        except Exception as exc:
            messagebox.showerror('Ishga tushirish xatosi',str(exc),parent=self.root)
            self.status.set('Xato: parametrlarni tekshiring.')

    def _watch_worker(self,proc):
        try:
            code=proc.wait()
            self.msgs.put(('finished',proc,code))
        except Exception as exc:
            self.msgs.put(('line',proc,f'Worker xatosi: {exc}\n'))
            self.msgs.put(('finished',proc,1))

    def _read_worker_log(self):
        if self.current_output is None or self.proc is None:
            return
        logfile = self.current_output/'run.log'
        if not logfile.exists():
            return
        try:
            with logfile.open('r',encoding='utf-8',errors='replace') as f:
                f.seek(self.log_offset)
                new_text=f.read()
                self.log_offset=f.tell()
            if new_text:
                self._message(new_text)
        except OSError:
            pass

    def _consume_queue(self):
        self._read_worker_log()
        try:
            while True:
                kind, proc, value=self.msgs.get_nowait()
                if proc is self.proc:
                    if kind=='line':self._message(str(value))
                    else:
                        self._read_worker_log()  # Drain final output before process reset.
                        self._on_finished(int(value))
        except queue.Empty:
            pass
        self.root.after(120,self._consume_queue)

    def _on_finished(self,code):
        self.proc=None
        self.spinner.stop()
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        if code==0:
            self.status.set(f'Hisoblash muvaffaqiyatli yakunlandi: {self.current_output}')
            self._message('\nYAKUN: tayyor.\n')
            self._load_report()
        else:
            self.status.set('Hisoblash to\'xtadi yoki xato yuz berdi. Jurnalni ko\'ring.')
            self._message(f'\nYAKUN: xato/to\'xtash (exit={code}).\n')

    def stop(self):
        if self.proc is not None:
            self._message('\nFoydalanuvchi to\'xtatishni so\'radi.\n')
            self.proc.terminate()
            self.status.set('To\'xtatilmoqda...')

    def _set_summary(self,text):
        self.summary.config(state='normal')
        self.summary.delete('1.0','end')
        self.summary.insert('1.0',text)
        self.summary.config(state='disabled')

    def _load_report(self):
        if not self.current_output:return
        base=self.current_output
        lines=[]
        for name in ('fsdt','clpt'):
            path=base/name/'summary.json'
            if not path.exists():continue
            try:
                dat=json.loads(path.read_text(encoding='utf-8'))
                lines.append('=== '+name.upper()+' ===')
                f=dat.get('instantaneous_frequencies_hz',[])
                lines.append('Oniy chastotalar [Hz]: '+', '.join(f'{float(x):.4g}' for x in f))
                f=dat.get('relaxed_frequencies_hz',[])
                lines.append('Relaksatsiyalangan [Hz]: '+', '.join(f'{float(x):.4g}' for x in f))
                if name=='fsdt':
                    peak=dat.get('peak',{})
                    if 'abs_w_m' in peak:
                        lines.append(f"Maks. |w|: {peak['abs_w_m']*1000:.5g} mm")
                    for r in dat.get('stress',[]):
                        lines.append(f"  {r.get('name')}: Mises {r.get('max_von_mises_pa',0)/1e6:.4g} MPa; "
                                     f"FI {r.get('max_failure_index')}")
                else:
                    if 'peak_center_deflection_m' in dat:
                        lines.append(f"Markaziy w: {dat['peak_center_deflection_m']*1000:.5g} mm")
            except (OSError, ValueError, KeyError,TypeError) as exc:
                lines.append('Natijani o\'qishda xato: '+str(exc))
        compare=base/'comparison.json'
        if compare.exists():
            dat=json.loads(compare.read_text(encoding='utf-8'))
            lines.append('=== TAQQOSLASH ===')
            lines.append('Maks. markaziy farq [mm]: '
                         f"{dat['center_max_abs_difference_mm']:.6g}")
        self._set_summary('\n'.join(lines) or 'Natijalar papkasini tekshiring.')
        self._draw_chart()

    def _chart_path(self):
        if not self.current_output:return None
        for path in (self.current_output/'comparison.png',
                     self.current_output/'fsdt'/'response.png',
                     self.current_output/'clpt'/'response.png'):
            if path.exists():return path
        # FSDT has different output plot names in older versions
        for path in (self.current_output/'fsdt').glob('*.png'):
            return path
        return None

    def _draw_chart(self):
        path=self._chart_path()
        if not path:
            return
        try:
            from PIL import Image, ImageTk
            w=max(self.graph_label.winfo_width()-12,400)
            h=max(self.graph_label.winfo_height()-12,240)
            with Image.open(path) as picture:
                img=picture.copy()
            img.thumbnail((w,h),Image.Resampling.LANCZOS)
            self._picture=ImageTk.PhotoImage(img)
            self.graph_label.config(image=self._picture,text='')
        except Exception as exc:
            self.graph_label.config(text=f'Grafik ko\'rsatilmadi: {exc}')

    def open_picture(self):
        path=self._chart_path()
        if not path:
            messagebox.showinfo('Grafik','Grafik hali mavjud emas.',parent=self.root)
        else:self._open_path(path)

    def open_output(self):
        if not self.current_output or not self.current_output.exists():
            messagebox.showinfo('Natijalar','Natijalar hali mavjud emas.',parent=self.root)
        else:self._open_path(self.current_output)

    @staticmethod
    def _open_path(path):
        if sys.platform=='win32':
            os.startfile(str(path))
        elif sys.platform=='darwin':
            subprocess.Popen(['open',str(path)])
        else:
            subprocess.Popen(['xdg-open',str(path)])

    def about(self):
        messagebox.showinfo('Dastur haqida',
            'CLPT + FSDT + Rjanitsin–Koltunov\n\n'
            'FSDT: MITC4 FEM (5 DOF)\nCLPT: Ritz usuli\n'
            'Viskoelastik qatlamlar va Winkler/Pasternak asosi.\n\n'
            'Chiziqli model. Muhandislik tasdiqlash va konvergentsiya talab etiladi.\n'
            'Hisoblash internet talab qilmaydi.',parent=self.root)

    def close(self):
        if self.proc is not None:
            if not messagebox.askyesno('Chiqish',
                                       'Hisoblash davom etmoqda. To\'xtatib chiqilsinmi?',
                                       parent=self.root):
                return
            self.proc.terminate()
        self.root.destroy()
