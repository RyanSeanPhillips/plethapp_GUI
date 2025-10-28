# Telemetry Setup Guide

PlethApp uses a dual telemetry system:
- **Google Analytics 4**: Usage statistics (unlimited events)
- **Sentry**: Crash reports with stack traces (5k events/month free)

Both services are free and take ~15 minutes to set up.

---

## Part 1: Google Analytics 4 Setup (~10 minutes)

### Step 1: Create GA4 Account
1. Go to https://analytics.google.com/
2. Click **"Start measuring"** (or sign in if you have account)
3. **Account setup:**
   - Account name: "PlethApp Analytics"
   - Click **Next**
4. **Property setup:**
   - Property name: "PlethApp"
   - Reporting timezone: Your timezone
   - Currency: USD (or your currency)
   - Click **Next**
5. **Business details:**
   - Industry: "Science & Education" or "Software"
   - Business size: "Small" (1-10 employees)
   - Click **Next**
6. **Business objectives:**
   - Select "Generate leads" or "Examine user behavior"
   - Click **Create**
7. Accept Terms of Service

### Step 2: Create Data Stream
1. Platform selection: Select **"Web"** (even though it's desktop app)
2. **Set up web stream:**
   - Website URL: `https://plethapp.local` (placeholder)
   - Stream name: "PlethApp Desktop"
   - Click **"Create stream"**
3. You'll see a screen with **"Measurement ID"** (format: `G-XXXXXXXXXX`)
   - **Copy this** - you'll need it in Step 4

### Step 3: Create API Secret
1. In the data stream settings, scroll down to **"Measurement Protocol API secrets"**
2. Click **"Create"**
3. **Create secret:**
   - Nickname: "PlethApp Desktop"
   - Click **"Create"**
4. You'll see a **"Secret value"** (long alphanumeric string)
   - **Copy this** - you'll need it in Step 4
   - ⚠️ **Important**: You can only see this once! Save it now.

### Step 4: Add Credentials to PlethApp
1. Open `core/telemetry.py` in a text editor
2. Find lines 33-34 (near top of file):
   ```python
   GA4_MEASUREMENT_ID = None  # Format: "G-XXXXXXXXXX"
   GA4_API_SECRET = None      # Create in GA4 admin panel
   ```
3. Replace with your actual values:
   ```python
   GA4_MEASUREMENT_ID = "G-ABC123XYZ"  # Your measurement ID
   GA4_API_SECRET = "your_secret_value_here"  # Your API secret
   ```
4. Save the file

### Step 5: Test GA4 Integration
1. Delete any existing config:
   - Windows: Delete `C:\Users\{yourname}\AppData\Roaming\PlethApp\`
2. Run PlethApp:
   ```bash
   python run_debug.py
   ```
3. Complete the welcome dialog
4. Load a file and close the app
5. Check GA4 dashboard (may take 1-2 minutes to appear):
   - Go to https://analytics.google.com/
   - Click **Reports** → **Realtime**
   - You should see 1 active user and events like `session_start`, `file_loaded`

✅ **GA4 Setup Complete!**

---

## Part 2: Sentry Setup (~5 minutes)

### Step 1: Create Sentry Account
1. Go to https://sentry.io/signup/
2. Sign up with Google/GitHub or email
3. **Create organization:**
   - Organization name: "PlethApp" or your name
   - Click **Continue**

### Step 2: Create Project
1. **Choose platform:** Select **"Python"**
2. **Alert frequency:** Default (recommended)
3. **Project name:** "plethapp"
4. Click **"Create Project"**

### Step 3: Get DSN
1. You'll see a setup page with code examples
2. Look for the **DSN** (Data Source Name)
   - Format: `https://abc123@o456.ingest.sentry.io/789`
3. **Copy the DSN** - you'll need it in Step 4
4. You can skip the rest of the setup (we already have code)

### Step 4: Add DSN to PlethApp
1. Open `core/telemetry.py` in a text editor
2. Find line 38:
   ```python
   SENTRY_DSN = None  # Format: "https://abc123@o456.ingest.sentry.io/789"
   ```
3. Replace with your actual DSN:
   ```python
   SENTRY_DSN = "https://abc123@o456.ingest.sentry.io/789"  # Your DSN
   ```
4. Save the file

### Step 5: Install Sentry SDK
```bash
pip install sentry-sdk
```

### Step 6: Test Sentry Integration
1. Run PlethApp:
   ```bash
   python run_debug.py
   ```
2. In the console, you should see:
   ```
   Telemetry: Sentry initialized for crash reports
   Telemetry: Initialized (GA4 + Sentry)
   ```
3. To test crash reporting, add this temporary code somewhere:
   ```python
   # Temporary test - remove after testing
   raise ValueError("Test crash report")
   ```
4. Check Sentry dashboard:
   - Go to https://sentry.io/
   - You should see the error appear within seconds
   - Click it to see stack trace

✅ **Sentry Setup Complete!**

---

## What Gets Tracked

### Google Analytics (Usage Statistics)
- ✅ Number of users (anonymous UUID)
- ✅ Files loaded (type: ABF/SMRX/EDF, count, no file names)
- ✅ Features used (GMM clustering, spectral analysis, etc.)
- ✅ Exports (PDF, CSV, NPZ counts)
- ✅ Session duration
- ✅ Platform (Windows/Mac/Linux)
- ✅ PlethApp version

### Sentry (Crash Reports)
- ✅ Error type (ValueError, IndexError, etc.)
- ✅ Stack trace (file names, line numbers)
- ✅ Platform and version info
- ❌ No file names or data
- ❌ No personal info

### Local Fallback (if network unavailable)
- Everything logs to: `C:\Users\{user}\AppData\Roaming\PlethApp\telemetry.log`
- JSON format, one event per line

---

## Viewing Analytics

### Google Analytics Dashboard
1. Go to https://analytics.google.com/
2. Select your property: "PlethApp"
3. **Realtime Report** (see active users right now):
   - Reports → Realtime
4. **User Report** (see total users):
   - Reports → User → User attributes
5. **Events Report** (see what users do):
   - Reports → Engagement → Events
   - See counts for: `file_loaded`, `feature_used`, `export`, etc.
6. **Custom Report** (create your own):
   - Explore → Create exploration
   - Add dimensions: `event_name`, `platform`, `version`
   - Add metrics: `event_count`, `active_users`

### Sentry Dashboard
1. Go to https://sentry.io/
2. Select project: "plethapp"
3. **Issues** (see all crashes):
   - Click issue to see stack trace
   - See how many users affected
   - Get email alerts for new issues
4. **Releases** (track errors by version):
   - Set up releases to track which version has most errors

---

## Privacy Controls

Users can disable telemetry anytime:

1. **In welcome dialog** (first launch):
   - Uncheck "Share anonymous usage statistics"
   - Uncheck "Send crash reports"

2. **In Help → About** (anytime):
   - Press F1 → About tab
   - Toggle checkboxes

3. **Manually** (delete config):
   - Delete `C:\Users\{user}\AppData\Roaming\PlethApp\config.json`

---

## Troubleshooting

### "Telemetry: Using local logging"
- GA4 credentials not configured → Edit `core/telemetry.py` lines 33-34

### "Telemetry: Sentry not installed"
- Run: `pip install sentry-sdk`

### "Telemetry: GA4 send failed"
- Network issue (normal) → Events logged locally to `telemetry.log`

### No events in GA4 dashboard
- Wait 1-2 minutes for data to appear
- Check Realtime report first
- Verify `GA4_MEASUREMENT_ID` and `GA4_API_SECRET` are correct

### No errors in Sentry dashboard
- Good! No crashes yet.
- Test with temporary `raise Exception("test")` code

---

## Cost Estimates

### Current Usage (0-100 users/month)
- **Google Analytics**: $0/month (free forever)
- **Sentry**: $0/month (free tier)
- **Total**: **$0/month**

### Growth Scenario (100-1000 users/month)
- **Google Analytics**: $0/month (still free)
- **Sentry**:
  - If <100 crashes/month: $0/month (still free)
  - If >100 crashes/month: $29/month (Team plan)

---

## Next Steps

After setup:
1. ✅ Test both integrations
2. ✅ Remove test crash code
3. ✅ Commit updated `core/telemetry.py`
4. ✅ Build executable and distribute
5. 📊 Check analytics weekly to see adoption

---

## Future Enhancements

Want more telemetry? Add these events:

```python
# In main.py

# Manual editing
def on_add_peak_clicked(self):
    telemetry.log_feature_used('manual_editing_add_peak')
    # ... existing code

# Spectral analysis
def on_spectral_analysis_clicked(self):
    telemetry.log_feature_used('spectral_analysis')
    # ... existing code

# Exports
def export_pdf(self):
    telemetry.log_export('summary_pdf')
    # ... existing code

def export_breaths_csv(self):
    telemetry.log_export('breaths_csv')
    # ... existing code
```

---

**Questions?** See `core/telemetry.py` for implementation details.
