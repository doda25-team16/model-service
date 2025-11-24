# model-service/Vagrantfile
NUMBER_OF_WORKERS = 2

Vagrant.configure("2") do |config|
  config.vm.box = "bento/ubuntu-24.04"
  config.vm.provider "virtualbox" do |vb|
    vb.gui = false
  end

  # Control Node
  config.vm.define "ctrl" do |ctrl|
    ctrl.vm.hostname = "ctrl"
    # "host-only" interface (NIC)
    ctrl.vm.network "private_network", ip: "192.168.56.100"
    
    ctrl.vm.provider "virtualbox" do |wvb| # wvb = worker virtual box
      wvb.memory = 4096
      wvb.cpus = 1
    end
  end

  # Worker Nodes
  (1..NUMBER_OF_WORKERS).each do |i|
    config.vm.define "node-#{i}" do |node|
      node.vm.hostname = "node-#{i}"
      # "host-only" interface (NIC)
      node.vm.network "private_network", ip: "192.168.56.10#{i}" # ip is indexed by worker#
      
      node.vm.provider "virtualbox" do |nvb| # nvb = worker virtual box
        nvb.memory = 6144
        nvb.cpus = 2
      end
    end
  end
end
