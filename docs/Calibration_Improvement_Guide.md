# Hand-Eye Calibration Improvement Guide

## Current Performance
- Position std dev: ~4 mm
- Max deviation: ~6 mm
- Status: ✅ Good, but can be improved

## Improvement Strategies

### 1. Data Collection Quality (Most Important)

#### 1.1 Increase Sample Diversity
**Current**: You may be using poses that are too similar.

**Recommendations**:
- Collect 15-30 samples (instead of 5-10)
- Ensure large variation in:
  - Robot position (X, Y, Z)
  - Robot orientation (RX, RY, RZ)
  - Distance to AprilTag (near and far)
  - Viewing angles (perpendicular and oblique)

**Avoid**:
- Degenerate motions (only translation, only rotation)
- Small movements between poses
- Always viewing tag from same angle

#### 1.2 Optimal Pose Distribution
```
Good pose set should include:
- 5-8 poses: Close to tag (300-400mm), different angles
- 5-8 poses: Medium distance (400-600mm), various orientations
- 5-8 poses: Far from tag (600-800mm), still clearly visible
- Mix of pure translations and combined rotation+translation
```

### 2. AprilTag Detection Accuracy

#### 2.1 Tag Size
- **Current**: Check your tag size (should be printed accurately)
- **Recommendation**: Use larger tag if possible (80mm - 150mm)
- Larger tags = more stable corner detection = better accuracy

#### 2.2 Lighting Conditions
- Uniform, diffuse lighting (avoid shadows)
- Avoid glare/reflections on tag surface
- Consider using LED panel lights

#### 2.3 Camera Settings
- Use manual focus (disable auto-focus)
- Lower ISO to reduce noise
- Ensure proper exposure (tag should be clearly visible)

### 3. Camera Intrinsic Calibration

**Critical**: Make sure your camera intrinsics are well calibrated!

Poor intrinsic calibration is a major source of error.

**Check**:
- When was the last camera calibration?
- Did you use enough calibration images (20-30)?
- Was the calibration board large enough?
- Did you cover the full field of view?

**Recommendation**: Recalibrate camera with high-quality checkerboard/ChArUco board.

### 4. Try Different Calibration Methods

Current: Using `tsai` method.

Try all methods and compare results:
```bash
# Try all methods
for method in tsai park horaud andreff daniilidis; do
  echo "Testing method: $method"
  python3 scripts/handeye_calibrate.py \
    --input-csv data/robot_eye_in_hand.csv \
    --method $method \
    --max-reproj-error 2.0 \
    --min-samples 10
done
```

Different methods work better with different data distributions.

### 5. Data Filtering

#### 5.1 Reprojection Error Threshold
- **Current**: 2.0 pixels
- **Recommendation**: Try 1.0 or 1.5 for higher quality data

#### 5.2 Remove Outliers
Modify the calibration script to:
- Calculate calibration with all samples
- Remove samples with high residuals
- Re-calibrate with filtered samples

### 6. Mechanical Improvements

#### 6.1 Camera Mounting
- Ensure camera is rigidly mounted to end-effector
- Check for any play or vibration
- Tighten all mounting screws

#### 6.2 Robot Stability
- Wait for robot to fully settle before capturing (add delay)
- Avoid high speeds during motion
- Check robot calibration status

#### 6.3 AprilTag Mounting
- Mount tag on rigid, flat surface
- Ensure tag is not bending or warped
- Use high-quality print (laser printer, not inkjet)

### 7. Temperature Stabilization

Camera sensors drift with temperature:
- Let camera warm up for 10-15 minutes
- Keep lab temperature stable
- Avoid direct sunlight on camera

### 8. Multi-Pose Validation

Add validation at more diverse poses:
```bash
python3 scripts/validate_handeye_calibration.py \
  --num-samples 10 \
  --manual \
  ...
```

Test at:
- Different distances
- Different angles
- Different robot configurations

### 9. Expected Accuracy Limits

**Theoretical limits** for your setup:
- AprilTag detection: ~0.5-1mm at 500mm distance
- Robot repeatability: ~0.1-0.5mm (RealMan spec)
- Combined error: ~1-2mm is excellent

**Your current 4mm** is already very good for practical applications!

### 10. Application-Specific Tuning

**If your application requires < 2mm accuracy**:
- Consider stereo camera setup
- Use larger/multiple tags
- Implement online calibration refinement
- Add visual servoing for final precision

## Recommended Next Steps

1. **Immediate** (Low effort, high impact):
   - Collect 20-30 diverse poses
   - Try different calibration methods
   - Lower reproj error threshold to 1.0

2. **Short-term** (Medium effort):
   - Recalibrate camera intrinsics
   - Improve lighting setup
   - Use larger AprilTag

3. **Long-term** (High effort):
   - Upgrade camera resolution
   - Implement multi-tag calibration
   - Add online calibration refinement

## Validation Script Enhancement

Collect data with better distribution:
- 5 poses at 300mm distance
- 5 poses at 500mm distance
- 5 poses at 700mm distance
- Each distance: vary orientation ±30° in all axes

This ensures calibration is robust across the workspace.
