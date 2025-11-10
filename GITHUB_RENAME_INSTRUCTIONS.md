# GitHub Repository Rename Instructions

## Renaming plethapp_GUI to PhysioMetrics

This guide walks you through renaming your GitHub repository from `plethapp_GUI` to `PhysioMetrics`.

---

## Step 1: Rename Repository on GitHub

1. **Go to your repository:**
   - Navigate to: https://github.com/RyanSeanPhillips/plethapp_GUI

2. **Open Settings:**
   - Click the **Settings** tab (gear icon in the top-right menu)

3. **Change Repository Name:**
   - Find the "Repository name" section at the top
   - Change `plethapp_GUI` to `PhysioMetrics`
   - Click **Rename**

4. **Confirm the rename:**
   - GitHub will show a confirmation dialog
   - Click **I understand, rename repository**

---

## Step 2: Update Local Git Remote

After renaming on GitHub, update your local repository's remote URL:

```bash
# Navigate to your local repository
cd "C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6"

# Update the remote URL
git remote set-url origin https://github.com/RyanSeanPhillips/PhysioMetrics.git

# Verify the change
git remote -v
```

Expected output:
```
origin  https://github.com/RyanSeanPhillips/PhysioMetrics.git (fetch)
origin  https://github.com/RyanSeanPhillips/PhysioMetrics.git (push)
```

---

## Step 3: What GitHub Handles Automatically

When you rename a repository, GitHub automatically:

- ✅ **Redirects old URLs** - Old links like `github.com/RyanSeanPhillips/plethapp_GUI` automatically redirect to the new name
- ✅ **Preserves local clones** - Your existing local repository continues to work without changes
- ✅ **Maintains forks and stars** - All forks, stars, watchers, and issues are preserved
- ✅ **Updates web interface** - All links in Issues, Pull Requests, and README are automatically updated

---

## Step 4: Commit and Push Rebranding Changes

Once the repository is renamed, commit all the rebranding changes:

```bash
# Add all changed files
git add -A

# Commit the changes
git commit -m "Rebrand application from PlethApp to PhysioMetrics

- Updated README.md, CHANGELOG.md, and CITATION.cff
- Updated all source code references
- Renamed pleth_app.spec to physiometrics.spec
- Updated build scripts and internal documentation
- Updated Help dialog with new branding and author information"

# Push to the renamed repository
git push origin main
```

---

## Step 5: Update References Elsewhere (Optional)

Consider updating references in:

1. **Documentation links** - Any external docs that link to the repository
2. **Personal website** - Update portfolio or CV links
3. **Paper drafts** - Update any manuscript references to the repository
4. **Bookmarks** - Update your browser bookmarks (though redirects work)
5. **CI/CD configs** - If using external services, update their webhook URLs

---

## Step 6: Zenodo DOI Setup (After Rename)

Once rebranding is complete, you can proceed with Zenodo DOI:

1. **Link GitHub to Zenodo:**
   - Go to: https://zenodo.org/account/settings/github/
   - Click "Sync now" to see your renamed repository
   - Enable the toggle for "PhysioMetrics"

2. **Create GitHub Release:**
   - Go to: https://github.com/RyanSeanPhillips/PhysioMetrics/releases/new
   - Tag version: `v1.0.11`
   - Release title: `PhysioMetrics v1.0.11`
   - Description: Copy content from CHANGELOG.md
   - Click "Publish release"

3. **Get DOI from Zenodo:**
   - Zenodo automatically detects the release
   - Wait a few minutes for processing
   - Visit: https://zenodo.org/account/settings/github/
   - Find your release and copy the DOI badge

4. **Update Files with Real DOI:**
   - Replace placeholder DOI in README.md
   - Replace placeholder DOI in CITATION.cff
   - Replace placeholder DOI in dialogs/help_dialog.py
   - Commit and push: `git commit -am "Add Zenodo DOI" && git push`

---

## Troubleshooting

### Problem: "Repository not found" error when pushing

**Solution:** Your local repository still points to the old URL. Run:
```bash
git remote set-url origin https://github.com/RyanSeanPhillips/PhysioMetrics.git
```

### Problem: Clone fails with "repository not found"

**Solution:** Update your clone command to use the new name:
```bash
git clone https://github.com/RyanSeanPhillips/PhysioMetrics.git
```

### Problem: Old URLs still work but I want to update them

**Reason:** GitHub provides permanent redirects from old repository names. This is by design and ensures no broken links.

---

## Summary Checklist

- [ ] Rename repository on GitHub (Settings → Repository name → PhysioMetrics)
- [ ] Update local git remote URL
- [ ] Verify remote URL with `git remote -v`
- [ ] Commit rebranding changes
- [ ] Push to renamed repository
- [ ] Update external references (documentation, website, etc.)
- [ ] Set up Zenodo DOI (after rename is complete)
- [ ] Update files with real Zenodo DOI

---

**Note:** You can safely delete this file after completing the rename process.
