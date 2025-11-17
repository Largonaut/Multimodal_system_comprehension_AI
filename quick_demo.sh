#!/bin/bash
# GREMLIN Quick Demo Setup
# Run this script to quickly test the full demo on one machine

set -e

echo "🚀 GREMLIN Quick Demo Setup"
echo "============================"
echo ""

# Find language pack
PACK=$(find language_packs -name "*.json" -type f | head -n 1)

if [ -z "$PACK" ]; then
    echo "📦 No language pack found. Generating one..."
    python generate_language_pack.py --words 500 --output language_packs/
    PACK=$(find language_packs -name "*.json" -type f | head -n 1)
fi

echo "✅ Using language pack: $PACK"
echo ""

# Check dependencies
echo "📋 Checking dependencies..."
python -c "import rich" 2>/dev/null || pip install rich -q
echo "✅ Dependencies OK"
echo ""

# Show options
echo "Choose demo mode:"
echo "  1) Three-terminal demo (best for rehearsal)"
echo "  2) Scripted demo (single terminal)"
echo "  3) Interactive client-server (manual control)"
echo ""

read -p "Choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🎬 Three-Terminal Demo"
        echo "====================="
        echo ""
        echo "Opening 3 terminal windows..."
        echo ""
        echo "Terminal 1: MITM Viewer"
        echo "Terminal 2: Server"
        echo "Terminal 3: Client"
        echo ""
        echo "Press any key to start..."
        read -n 1

        # Try to open in separate terminals based on OS
        if command -v gnome-terminal &> /dev/null; then
            gnome-terminal -- bash -c "python demo/mitm_viewer.py --mode passive; bash"
            sleep 1
            gnome-terminal -- bash -c "python demo/server.py --pack '$PACK'; bash"
            sleep 1
            gnome-terminal -- bash -c "python demo/client.py --pack '$PACK' --mode scripted --rounds 5; bash"
        elif command -v xterm &> /dev/null; then
            xterm -hold -e "python demo/mitm_viewer.py --mode passive" &
            sleep 1
            xterm -hold -e "python demo/server.py --pack '$PACK'" &
            sleep 1
            xterm -hold -e "python demo/client.py --pack '$PACK' --mode scripted --rounds 5" &
        else
            echo "❌ Could not detect terminal emulator"
            echo "Please open 3 terminals manually and run:"
            echo ""
            echo "Terminal 1: python demo/mitm_viewer.py --mode passive"
            echo "Terminal 2: python demo/server.py --pack $PACK"
            echo "Terminal 3: python demo/client.py --pack $PACK --mode scripted --rounds 5"
        fi
        ;;

    2)
        echo ""
        echo "🎬 Scripted Demo (no network needed)"
        echo "===================================="
        echo ""
        python demo_authentication.py --pack "$PACK" --rounds 5
        ;;

    3)
        echo ""
        echo "🎬 Interactive Client-Server Demo"
        echo "=================================="
        echo ""
        echo "Start server in another terminal:"
        echo "  python demo/server.py --pack $PACK"
        echo ""
        read -p "Press Enter when server is ready..."
        echo ""
        python demo/client.py --pack "$PACK" --mode interactive
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Demo complete!"
