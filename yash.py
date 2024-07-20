# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.shortcuts import render
import pandas as pd
from datetime import datetime
from forms import UploadFileForm, TowerForm
from django.conf import settings
import os
import pandas as pd





def process_excel(path_excel):
    ant_layout = pd.read_excel(path_excel, sheet_name="Antenna Layout",index_col=0,header=None)
    site_id = ant_layout.iloc[0,4]
    scan_date = ant_layout.iloc[1,4]
    mount_level = ant_layout.iloc[2,4]
    report_version = ant_layout.iloc[1,11]

    ant_layout_design = pd.read_excel(path_excel, sheet_name="Antenna Layout",usecols='A:AA',header=32,index_col=None)
    ant_layout_design = ant_layout_design.dropna(axis=1, how='all')
    ant_layout_design = ant_layout_design.iloc[1:13,:]
    ant_layout_design = ant_layout_design.dropna(axis = 0, how = 'all')

    ant_layout_built = pd.read_excel(path_excel, sheet_name="Antenna Layout",usecols='AE:BE',header=32,index_col=None)
    ant_layout_built = ant_layout_built.dropna(axis=1, how='all')
    ant_layout_built = ant_layout_built.iloc[1:13,:]
    ant_layout_built = ant_layout_built.dropna(axis = 0, how = 'all')

    ant_swing = pd.read_excel(path_excel, sheet_name="Antenna Swing",usecols='Y:AU',header=5,index_col=None)
    ant_swing = ant_swing.dropna(axis=1, how='all')
    ant_swing = ant_swing.iloc[1:19,:]
    ant_swing = ant_swing.drop(columns=['Unnamed: 44'])
    ant_swing = ant_swing.dropna(axis = 0, how = 'all')

    min_swing = pd.read_excel(path_excel, sheet_name="Antenna Swing",usecols='Y:AU',header=24,index_col=None)
    min_swing = min_swing.iloc[0,20]

    sec_a = pd.read_excel(path_excel, sheet_name="Mounts",usecols='AC:AP',header=24,index_col=None)
    sec_a = sec_a.dropna(axis=1, how='all')
    sec_a = sec_a.iloc[0:6,:]
    sec_a = sec_a.dropna(axis = 0, how = 'all')
    sec_a = sec_a.drop(columns=['Unnamed: 39'])

    sec_b = pd.read_excel(path_excel, sheet_name="Mounts",usecols='AC:AP',header=43,index_col=None)
    sec_b = sec_b.dropna(axis=1, how='all')
    sec_b = sec_b.iloc[0:6,:]
    sec_b = sec_b.dropna(axis = 0, how = 'all')
    sec_b = sec_b.drop(columns=['Unnamed: 39'])

    sec_c = pd.read_excel(path_excel, sheet_name="Mounts",usecols='AC:AP',header=63,index_col=None)
    sec_c = sec_c.dropna(axis=1, how='all')
    sec_c = sec_c.iloc[0:6,:]
    sec_c = sec_c.dropna(axis = 0, how = 'all')
    sec_c = sec_c.drop(columns=['Unnamed: 39'])

    sec_a_dim = pd.read_excel(path_excel, sheet_name="Mounts",usecols='AC:AF',header=32,index_col=None)
    sec_a_dim = sec_a_dim.dropna(axis=1, how='all')
    sec_a_dim = sec_a_dim.iloc[0:8,:]
    sec_a_dim = sec_a_dim.dropna(axis = 0, how = 'all')

    sec_b_dim = pd.read_excel(path_excel, sheet_name="Mounts",usecols='AC:AF',header=51,index_col=None)
    sec_b_dim = sec_b_dim.dropna(axis=1, how='all')
    sec_b_dim = sec_b_dim.iloc[0:8,:]
    sec_b_dim = sec_b_dim.dropna(axis = 0, how = 'all')

    sec_c_dim = pd.read_excel(path_excel, sheet_name="Mounts",usecols='AC:AF',header=71,index_col=None)
    sec_c_dim = sec_c_dim.dropna(axis=1, how='all')
    sec_c_dim = sec_c_dim.iloc[0:8,:]
    sec_c_dim = sec_c_dim.dropna(axis = 0, how = 'all')

    ant_mount = pd.read_excel(path_excel, sheet_name="Mounts",usecols='AC:AP',header=5,index_col=None)
    ant_mount = ant_mount.iloc[0:2,:]
    ant_mount = ant_mount.dropna(axis=1, how='all')

    return site_id,scan_date,mount_level,report_version,ant_layout_design,ant_layout_built,ant_swing,min_swing,sec_a,sec_b,sec_c,sec_a_dim,sec_b_dim,sec_c_dim,ant_mount



@login_required(login_url="/login/")
def index(request):
    if request.method == "POST":
        context = {'segment': 'index'}
        site_id,scan_date,mount_level,report_version,ant_layout_design,ant_layout_built,ant_swing,min_swing,sec_a,sec_b,sec_c,sec_a_dim,sec_b_dim,sec_c_dim,ant_mount = process_excel()
        context.update({
            'site_id': site_id,
            'scan_date': scan_date.date(),
            'mount_level': mount_level,
            'report_version': report_version,
            'ant_layout_design': ant_layout_design.to_html(classes='table table-hover'),
            'ant_layout_built': ant_layout_built.to_html(classes='table table-hover'),
            'ant_swing': ant_swing.to_html(classes='table table-hover'),
            'min_swing': min_swing,
            'sec_a': sec_a.to_html(classes='table table-hover'),
            'sec_b': sec_b.to_html(classes='table table-hover'),
            'sec_c': sec_c.to_html(classes='table table-hover'),
            'sec_a_dim': sec_a_dim.to_html(classes='table table-hover'),
            'sec_b_dim': sec_b_dim.to_html(classes='table table-hover'),
            'sec_c_dim': sec_c_dim.to_html(classes='table table-hover'),
            'ant_mount': ant_mount.to_html(classes='table table-hover'),
        })
        html_template = loader.get_template('home/index.html')
        return HttpResponse(html_template.render(context, request))
    else:
        form = UploadFileForm()
        return render(request, "home/index.html", {"form": form})

@login_required(login_url="/login/")
def upload(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            tower = form.save()
            tower_excel = tower.excel_file.path
            tower_pdf = tower.pdf_file.path


            return render(request, "home/upload.html", {"form": form})
    else:
        form = UploadFileForm()
    return render(request, "home/upload.html", {"form": form})


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))