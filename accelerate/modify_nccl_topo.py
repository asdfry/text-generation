import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pci_speed", type=int, required=True)
parser.add_argument("-n", "--nic_speed", type=int, required=True)
parser.add_argument("-s", "--gpu_sm", type=int, required=True)
args = parser.parse_args()

pattern_pci = re.compile(r"\d+\.\d GT/s")
pattern_nic = re.compile(r"speed=\"\d+\"")
pattern_sm = re.compile(r"sm=\"\d+\"")

with open("./nccl-topo.xml", "r") as f:
    xml = f.read()

xml = pattern_pci.sub(f"{args.pci_speed}.0 GT/s", xml)
xml = pattern_nic.sub(f'speed="{args.nic_speed}"', xml)
xml = pattern_sm.sub(f'sm="{args.gpu_sm}"', xml)

with open("./nccl-topo.xml", "w") as f:
    f.write(xml)
