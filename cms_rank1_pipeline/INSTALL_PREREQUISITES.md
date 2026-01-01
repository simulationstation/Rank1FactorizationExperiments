# Prerequisites Installation Guide

**Detected OS:** Ubuntu 24.04

Both **CVMFS** and **Docker** are required but missing. Install them as follows:

---

## 1. Install Docker

### Ubuntu/Debian

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc 2>/dev/null || true

# Install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group (to run without sudo)
sudo usermod -aG docker $USER

# Verify installation
sudo docker run hello-world
```

### Fedora/RHEL

```bash
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

---

## 2. Install CVMFS

### Ubuntu/Debian

```bash
# Install CVMFS package
wget https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest_all.deb
sudo dpkg -i cvmfs-release-latest_all.deb
rm -f cvmfs-release-latest_all.deb
sudo apt-get update
sudo apt-get install -y cvmfs cvmfs-config-default

# Configure CVMFS
sudo bash -c 'cat > /etc/cvmfs/default.local << EOF
CVMFS_REPOSITORIES=cms.cern.ch,cms-ib.cern.ch
CVMFS_HTTP_PROXY=DIRECT
CVMFS_QUOTA_LIMIT=20000
EOF'

# Setup and probe
sudo cvmfs_config setup
sudo cvmfs_config probe cms.cern.ch

# Verify
ls /cvmfs/cms.cern.ch/
```

### Fedora/RHEL/Arch

```bash
# Fedora/RHEL
sudo yum install -y https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest.noarch.rpm
sudo yum install -y cvmfs cvmfs-config-default

# Configure
sudo bash -c 'cat > /etc/cvmfs/default.local << EOF
CVMFS_REPOSITORIES=cms.cern.ch
CVMFS_HTTP_PROXY=DIRECT
CVMFS_QUOTA_LIMIT=20000
EOF'

sudo cvmfs_config setup
sudo cvmfs_config probe cms.cern.ch

# Arch (via AUR)
yay -S cvmfs
```

---

## 3. After Installation

**IMPORTANT:** After installing Docker, you need to log out and log back in (or run `newgrp docker`) for group membership to take effect.

Then verify both are working:

```bash
# Test Docker
docker run hello-world

# Test CVMFS
ls /cvmfs/cms.cern.ch/cmsset_default.sh
```

---

## 4. Re-run the Pipeline

Once both are installed, re-run this task and I will:
1. Build the Docker image
2. Run the containerized CMSSW workflow
3. Produce the CSV spectra and rank-1 test results

---

## Alternative: Use Singularity/Apptainer (No Root Required)

If you cannot install CVMFS on the host, you can use a Singularity container that bundles CVMFS:

```bash
# Install Apptainer (successor to Singularity)
sudo apt-get install -y apptainer

# Pull CMS CVMFS container
apptainer pull docker://cmssw/cms:rhel7

# Run with CVMFS
apptainer shell --bind /work cms_rhel7.sif
source /cvmfs/cms.cern.ch/cmsset_default.sh
```

---

## WSL2 Note

If running on WSL2 (Windows Subsystem for Linux):
- Docker Desktop for Windows can provide Docker
- CVMFS may require additional configuration or the CVMFS-CSI approach
- Consider using the Singularity alternative above
