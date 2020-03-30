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

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DeepFocusGUIClient {

    private final ManagedChannel _channel;
    public final DeepFocusGrpc.DeepFocusStub _asyncStub;

    public DeepFocusGUIClient(String host, int port) {
        this(ManagedChannelBuilder.forAddress(host, port).usePlaintext().build());

        Logger.getLogger(DeepFocusGUI.class
                .getName()).log(Level.INFO, "DeepFocusGUIClient is successfully connected !");

    }

    DeepFocusGUIClient(ManagedChannel channel) {
        _channel = channel;
        _asyncStub = DeepFocusGrpc.newStub(channel);
    }

    public void shutdown() throws InterruptedException {
        _channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }
}
