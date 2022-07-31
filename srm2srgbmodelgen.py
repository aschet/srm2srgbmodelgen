# Beer Color SRM/EBC to sRGB Model Generator
# Copyright 2022 Thomas Ascher
# sRGB value generation was partially derived from an implementation by Thomas Mansencal
# SPDX-License-Identifier: MIT
#
# For more information about the computational methods see:
# https://www.homebrewtalk.com/threads/on-the-calculation-of-srm-rgb-values-in-the-srgb-color-space.413581
# https://www.homebrewtalk.com/threads/ebc-or-srm-to-color-rgb.78018
# https://stackoverflow.com/questions/58722583/how-do-i-convert-srm-to-lab-using-e-308-as-an-algorithm
# deLange, A.J. (2016). Color. Brewing Materials and Processes. Elsevier. https://doi.org/10.1016/b978-0-12-799954-8.00011-3
#
# The following dependencies are required: numpy, matplotlib, colour-science, pandas
# Run within Jupyter Notebook for better visualisation

import math
import numpy as np
import matplotlib.pyplot as plt
import colour
import colour.plotting
import pandas as pd

# Adjust the following constants to alter the model generation
GLASS_DIAMETER_CM = 5
CIE_OBSERVER_NAME = 'CIE 1931 2 Degree Standard Observer'
CIE_ILLUMINANT_NAME = 'D65'
USE_EBC_SCALE = False
MAX_SCALE_VALUE = 50
SCALE_STEP = 0.25
POLYFIT_DEGREE_R = 3
POLYFIT_DEGREE_G = 3
POLYFIT_DEGREE_B = 3
PRINT_HEX_TABLE = False

observer = colour.MSDS_CMFS[CIE_OBSERVER_NAME]
illuminant = colour.SDS_ILLUMINANTS[CIE_ILLUMINANT_NAME]
illuminant_xy = colour.CCS_ILLUMINANTS[CIE_OBSERVER_NAME][CIE_ILLUMINANT_NAME]

if USE_EBC_SCALE == True:
    unit_name = 'EBC'
    unit_conversion = 1 / 1.97
else:
    unit_name = 'SRM'
    unit_conversion = 1.0

def poly_const_to_text(val):
    return '{:.4e}'.format(val)

def poly_to_text(coeff, input_name):
    text=''
    for i in reversed(coeff[1:]):
        text += poly_const_to_text(i) + '+' + input_name + '*('
    text += poly_const_to_text(coeff[0])
    text = text + ')' * (len(coeff) - 1)
    return text

def compile_poly(coeff, input_name):
    poly_text = poly_to_text(coeff, input_name.lower())
    code = compile(poly_text, 'model', 'eval')
    return poly_text, code

def eval_poly(code, scale):
    if USE_EBC_SCALE == True:
        ebc = scale
        return eval(code)
    else:
        srm = scale
        return eval(code)

def fit_poly(channel, degree):
    idx = np.isfinite(channel)
    return np.polyfit(scale_fit[idx], channel[idx], degree)

def clip_signal(signal):
    signal[(signal < 0.0) | (signal > 1.0)] = np.nan
    return signal

def clip_channel(rgb, channel):
    return clip_signal(np.array([i[channel] for i in rgb]))

def generate_rgb_data(scale):
    rgb = []
    wl = colour.SpectralShape(380, 780, 5).range()
    for i in scale:
        srm = i * unit_conversion
        values = 10**(-(srm / 12.7) * (0.02465 * math.e**(-(wl - 430.0) / 17.591) + 0.97535 * math.e**(-(wl - 430.0) / 82.122)) * GLASS_DIAMETER_CM)
        xyz = colour.sd_to_XYZ(colour.SpectralDistribution(values, wl), cmfs=observer, illuminant=illuminant) / 100.0
        rgb.append(colour.XYZ_to_sRGB(xyz, illuminant=illuminant_xy))
    return rgb

# Generate sRGB input data for model fit
scale_fit = np.arange(start=0, stop=MAX_SCALE_VALUE+SCALE_STEP,step=SCALE_STEP)
scale_display = np.arange(start=0, stop=MAX_SCALE_VALUE+1,step=1)
rgb_fit = generate_rgb_data(scale_fit)

# Fit data
r = clip_channel(rgb_fit, 0)
r_coeff = fit_poly(r, POLYFIT_DEGREE_R)
g = clip_channel(rgb_fit, 1)
g_coeff = fit_poly(g, POLYFIT_DEGREE_G)
b = clip_channel(rgb_fit, 2)
b_coeff = fit_poly(b, POLYFIT_DEGREE_B)

# Generate and compile model code
r_text, r_code = compile_poly(r_coeff, unit_name)
g_text, g_code = compile_poly(g_coeff, unit_name)
b_text, b_code = compile_poly(b_coeff, unit_name)

# Print model
print('# ' + unit_name + ' to sRGB model, multiply outputs by 255 and clip between 0 and 255')
print('# ' + str(GLASS_DIAMETER_CM) + ' cm transmission, ' +  CIE_OBSERVER_NAME + ', ' + CIE_ILLUMINANT_NAME + ' illuminant')
print('r=' + r_text)
print('g=' + g_text)
print('b=' + b_text)

# Plot figures
rgb_display = generate_rgb_data(scale_display)
rgb_display_model = []
for i in zip(eval_poly(r_code, scale_display), eval_poly(g_code, scale_display), eval_poly(b_code, scale_display)):
    rgb_display_model.append(np.array(i))
rgb_display = rgb_display + rgb_display_model


fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(i, 0, 1)) for i in rgb_display], columns=len(scale_display), **{'standalone': False})
ax_scale.xaxis.set_label_text(unit_name)
ax_scale.xaxis.set_ticks_position('bottom')

fig_model = plt.figure()
ax_model = fig_model.subplots(1)
ax_model.xaxis.set_label_text(unit_name)
ax_model.yaxis.set_label_text('Relative Intensity')

def plot_channel(values, new_values, color, label):
    ax_model.plot(scale_fit, new_values, color=color, label=label + ' Fit', linestyle=':')    
    ax_model.plot(scale_fit, values, color=color, label=label)

r_new = clip_signal(eval_poly(r_code, scale_fit))
g_new = clip_signal(eval_poly(g_code, scale_fit))
b_new = clip_signal(eval_poly(b_code, scale_fit))

plot_channel(r, r_new, '#ff0000', 'R')
plot_channel(g, g_new, '#00ff00', 'G')
plot_channel(b, b_new, '#0000ff', 'B')
ax_model.legend()

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(list(np.round(np.clip(rgb * 255.0, 0.0, 255.0)).astype(int)))

def rgb_list_to_hex(rgb):
    return [rgb_to_hex(i) for i in rgb]

if PRINT_HEX_TABLE == True:
    table = pd.DataFrame({unit_name: scale_display, "Color": rgb_list_to_hex(rgb_display_model)})
    print(table)
