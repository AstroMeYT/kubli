# Kubli AI

Welcome to Kubli, a FOSS multi-system AI platform to bring free AI to the public. This service works a lot like BOINC. People can register their computers as "Workers", which host an AI server for others to use while you aren't using your computer. This allows for users of Kubli to access multiple open-weight models like it is a normal AI service.

## How to Use Kubli

By going to [this website](https://astromeyt.github.io/kubli), you can access the Kubli network for free. The entire backend code is available in this repository. The frontend supports multiple chats, memory, and document/image uploads. Please do expect bugs, and DO NOT input personal information into these AI models. If you are using someone else's instance, we cannot verify that they used the same ```worker.py``` as the one provided. These instances could be malicious.

## How to Set Up a Worker

To set up a worker, please have Ollama and Python installed on your system, and have a model downloaded that runs decently fast on your system. Download the ```worker.py``` file, and execute it with

``` bash

python3 worker.py

```

This will prompt you to choose the model you want to host. This will only show models you have installed. Once selected, it will prompt you for the server's URL. Our server's URL can be found in the ```url.txt``` file in this repository. After entering the URL, your system is connected and ready to receive prompts from users!

## How to Set Up a Server

The best way to set up your own server for Kubli is to download and run the ```server.py``` file, and point a tunnel or port forwarding configuration to ```YOUR_IP:13500```. Users and workers can then connect to that server.
