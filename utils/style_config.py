import tkinter as tk
from tkinter import ttk
import platform


def apply_styling(root):
    """Apply consistent styling to the application"""
    # Create a theme
    style = ttk.Style()

    # Use native theme as base
    system = platform.system()
    if system == "Windows":
        style.theme_use('vista')
    elif system == "Darwin":  # macOS
        style.theme_use('aqua')
    else:  # Linux and others
        style.theme_use('clam')

    # Configure common elements
    style.configure('TButton', padding=6)
    style.configure('TLabel', padding=3)
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TNotebook', tabposition='n')

    # Configure specific elements
    style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
    style.configure('Section.TLabel', font=('Arial', 11, 'bold'))

    # Configure colors
    root.option_add('*background', '#f0f0f0')
    root.option_add('*Button*background', '#e1e1e1')
    root.option_add('*Button*activeBackground', '#d1d1d1')

    # Set padding
    root.option_add('*padX', 5)
    root.option_add('*padY', 5)

    return style


def create_hover_style(style):
    """Create hover effects for buttons"""
    # Define hover effect for standard buttons
    style.map('TButton',
              background=[('active', '#d1d1d1'),
                          ('hover', '#e9e9e9')])

    # Define hover effect for accent buttons
    style.configure('Accent.TButton',
                    background='#4a7abd',
                    foreground='white',
                    padding=6)
    style.map('Accent.TButton',
              background=[('active', '#3a6aad'),
                          ('hover', '#5a8acd')])

    # Define hover effect for danger buttons
    style.configure('Danger.TButton',
                    background='#d9534f',
                    foreground='white',
                    padding=6)
    style.map('Danger.TButton',
              background=[('active', '#c9433f'),
                          ('hover', '#e9635f')])