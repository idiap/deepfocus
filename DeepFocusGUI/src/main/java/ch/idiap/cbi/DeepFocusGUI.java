/**
Code for the PyTorch implementation of
"DeepFocus: a Few-Shot Microscope Slide Auto-Focus using a Sample-invariant CNN-based Sharpness Function"

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Adrian Shajkofci <adrian.shajkofci@idiap.ch>,
All rights reserved.

This file is part of DeepFocus.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of mosquitto nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/


package ch.idiap.cbi;

import org.micromanager.MenuPlugin;
import org.micromanager.Studio;

import org.scijava.plugin.Plugin;
import org.scijava.plugin.SciJavaPlugin;

@Plugin(type = MenuPlugin.class)
public class DeepFocusGUI implements SciJavaPlugin, MenuPlugin {

    private Studio _studio;
    private DeepFocusGUIFrame _frame;

    /**
     * This method receives the Studio object, which is the gateway to the
     * Micro-Manager API. You should retain a reference to this object for the
     * lifetime of your plugin. This method should not do anything except for
     * store that reference, as Micro-Manager is still busy starting up at the
     * time that this is called.
     */
    @Override
    public void setContext(Studio studio) {
        _studio = studio;
    }

    /**
     * This method is called when your plugin is selected from the Plugins menu.
     * Typically at this time you should show a GUI (graphical user interface)
     * for your plugin.
     */
    @Override
    public void onPluginSelected() {
        if (_frame == null) {
            // We have never before shown our GUI, so now we need to create it.
            _frame = new DeepFocusGUIFrame(_studio);
            _studio.events().registerForEvents(_frame);
        }
        _frame.setVisible(true);
    }

    /**
     * This string is the sub-menu that the plugin will be displayed in, in the
     * Plugins menu.
     */
    @Override
    public String getSubMenu() {
        return "Developer Tools";
    }

    /**
     * The name of the plugin in the Plugins menu.
     */
    @Override
    public String getName() {
        return "DeepFocus";
    }

    @Override
    public String getHelpText() {
        return "DeepFocus Help";
    }

    @Override
    public String getVersion() {
        return "1.0";
    }

    @Override
    public String getCopyright() {
        return "Idiap Research Institute, 2019";
    }
}
